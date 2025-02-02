import datasets
import grain.python as grain
from input_pipeline import _input_pipeline_utils
import multihost_dataloading
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from functools import partial
import librosa
import jax
from jax import numpy as jnp
import numpy as np
import dac_jax
import os
import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordWriter
import tiktoken
#from datasets import disable_caching
from collections import defaultdict
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("/tmp/jax_cache")
# disable_caching()
#os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"]=str(1024*1024*1024*64)

DEVICE = "tpu"
MAX_LENGTH_AUDIO = 30 * 44100
MAX_LENGTH_TEXT = 10000
PER_DEVICE_BATCH_SIZE = 4
#GLOBAL_BATCH_SIZE = PER_DEVICE_BATCH_SIZE * jax.device_count()
SOURCE_SAMPLERATE = 16000
class HFParseAudioFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""
  def map(self, features):
    audio_44k = librosa.resample(features["audio"]["array"], orig_sr=SOURCE_SAMPLERATE, target_sr=44100)
    return {
        "audio": np.asarray(audio_44k, dtype=np.float32),
        "text": np.asarray(features["text"], dtype=np.int32),
        "speaker" : np.asarray(features["speaker"],dtype=np.int32)
    }   

class PadToMaxLength(grain.MapTransform):

  def map(self, data):
    audio_length = data["audio"].shape[0]
    padded_audio = np.pad(data["audio"],(0,MAX_LENGTH_AUDIO - data["audio"].shape[0]))
    text_length = data["text"].shape[0]
    padded_text = np.pad(data["text"],(0,MAX_LENGTH_TEXT - data["text"].shape[0]))
    return {
        "audio": padded_audio,
        "audio_length":audio_length,
        "text": padded_text,
        "text_length":text_length,
        "speaker": data["speaker"]
    }
if __name__ == "__main__":
    if DEVICE == "tpu":
        jax.distributed.initialize()
        device_mesh = mesh_utils.create_device_mesh((jax.device_count(), 1))
    else:
        device_mesh = mesh_utils.create_device_mesh((1, 1))
    mesh = Mesh(device_mesh, axis_names=("data", "model")) 
    dataset = datasets.load_dataset(
        "parler-tts/mls_eng",
        split="train",
        streaming=True,
    )
    
    cl100k_base = tiktoken.get_encoding("cl100k_base")

    enc = tiktoken.Encoding(
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|im_start|>": 100264,
            "<|im_end|>": 100265,
            "<|semantic|>": 100266,
        }
    )
    
    def process(example):
        ids = enc.encode(text=example["transcript"])
        
        return {'input_ids': ids}
    dataset = dataset.map(process)

    def get_sharding_for_spec(pspec: PartitionSpec) -> NamedSharding:
        """
        Get a NamedSharding for a given PartitionSpec, and the device mesh.
        A NamedSharding is simply a combination of a PartitionSpec and a Mesh instance.
        """
        return NamedSharding(mesh, pspec)
    model, variables = dac_jax.load_model(model_type="44khz")
    x_sharding = get_sharding_for_spec(PartitionSpec("data"))
    replicate_sharding = get_sharding_for_spec(PartitionSpec(None))
    @partial(jax.jit, in_shardings=x_sharding,out_shardings=replicate_sharding)
    def encode_to_codes(x: jnp.ndarray):
        codes, scale = model.apply(
            variables,
            x,
            method="encode",
        )
        return codes
    dataset = dataset.select_columns(["input_ids","audio","speaker_id"]).rename_column("input_ids", "text").rename_column("speaker_id", "speaker")
    dataset = _input_pipeline_utils.HFDataSource(dataset,
                                                0,
                                                1,
                                                1,
                                                False,
                                                15000,
                                                "text")
    operations = []
    operations.append(HFParseAudioFeatures())
    operations.append(PadToMaxLength())
    operations.append(grain.Batch(batch_size=PER_DEVICE_BATCH_SIZE * jax.device_count() // jax.process_count(), drop_remainder=True))
    dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=0, shard_count=1, drop_remainder=True
      ),
      shuffle=False,
      seed=0,
    )

    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=dummy_index_sampler,
        worker_count=1,  # only supports one worker for now, more workers results in duplicated data
        worker_buffer_size=1,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=128),
    )
    

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, mesh)

    CODEBOOK_PAD_TOKEN_ID = 0
    MAX_TOKEN_LENGTH = 8000
    i = 0
    writer = None
    speaker_semantic_dict = defaultdict(list)
    speaker_token_dict = defaultdict(list)
    os.makedirs("/dev/shm/dac_dataset_1",exist_ok=True)
    for item in multihost_gen:
        print(f"round {i}")
        if jax.process_index() == 0:
            if i%10240 == 0:
                num = i//10240
                if writer is not None:
                    writer.close() 
                writer = ArrayRecordWriter(f"/dev/shm/dac_dataset_1/mls_eng_train_part_{num}.arrayrecord", 'group_size:1')
            
        semantics = encode_to_codes(jnp.expand_dims(item["audio"],1))
        i+=1
        #semantics = jnp.asarray(semantics)
        text_lengths = jax.device_put(item["text_length"],replicate_sharding)
        n_frames = jax.device_put(item["audio_length"],replicate_sharding)
        text_tokens = jax.device_put(item["text"],replicate_sharding)
        speaker_ids = jax.device_put(item["speaker"],replicate_sharding)
        
        for k in range(PER_DEVICE_BATCH_SIZE * jax.device_count()):
            #print("working")
            n_frame = n_frames[k]//512
            text_length = text_lengths[k]
            text_token = text_tokens[k][:text_length]
            semantics_slice = semantics[k][:,:n_frame]
            speaker_id = int(speaker_ids[k])

            speaker_semantic_list = speaker_semantic_dict[speaker_id]
            speaker_token_list = speaker_token_dict[speaker_id]

            new_semantic_length = semantics_slice.shape[1]
            new_text_length = text_token.shape[0]

            semantics_slice = np.asarray(semantics_slice)
            text_slice = np.asarray(text_token)


            if sum(s.shape[1] for s in speaker_semantic_list) + sum(s.shape[0] for s in speaker_token_list) + new_semantic_length + new_text_length <= MAX_TOKEN_LENGTH:
                speaker_token_list.append(text_slice)
                speaker_semantic_list.append(semantics_slice)
            else:
                temp_text_slice = []
                for s in speaker_token_list:
                    if len(temp_text_slice) == 0:
                        temp_text_slice.extend(s.tolist())
                    else:
                        temp_text_slice.extend([enc.encode_single_token(" ")])
                        temp_text_slice.extend(s.tolist())
                temp_semantic_slice = None
                for s in speaker_semantic_list:
                    if temp_semantic_slice is None:
                        temp_semantic_slice = s
                    else:
                        temp_semantic_slice = np.concatenate((temp_semantic_slice,s),axis=1)
                string_prefix = "<|im_start|>user\n"
                string_suffix = "<|im_end|><|im_start|>assistant\n"

                encoded_prefix = enc.encode(
                    string_prefix,
                    allowed_special={"<|im_start|>","<|im_end|>"}
                )

                encoded_suffix = enc.encode(
                    string_suffix,
                    allowed_special={"<|im_start|>","<|im_end|>"}
                )


                encoded = encoded_prefix + temp_text_slice + encoded_suffix
                codebook_dim = 9

                semantic_token_id = enc.encode_single_token("<|semantic|>")
                semantic_length = temp_semantic_slice.shape[1]
                tokens = (
                    encoded
                    + [semantic_token_id] * semantic_length
                    + [enc.encode_single_token("<|im_end|>")]
                )
                prompt_length = len(encoded)

                
                codes = np.pad(temp_semantic_slice,((0,0),(prompt_length,1)),constant_values=CODEBOOK_PAD_TOKEN_ID)
                tokens = np.asarray(tokens)
                codes = codes.transpose(1,0)
                tokens = np.concatenate((tokens[...,np.newaxis],codes),axis=-1)
                
                
                example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'tokens': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tokens).numpy()]))
                            }
                        )
                    )
                if jax.process_index() == 0:
                    writer.write(example.SerializeToString())

                speaker_semantic_dict[speaker_id] = [semantics_slice]
                speaker_token_dict[speaker_id] = [text_slice]
    # writer.close() 