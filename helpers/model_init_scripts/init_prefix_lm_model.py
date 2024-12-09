import argparse
import os

from transformers import AutoConfig, AutoTokenizer

from parler_tts import PrefixLMParlerTTSDecoderConfig, PrefixLMParlerTTSForCausalLM, PrefixLMParlerTTSForConditionalGeneration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_directory", type=str, help="Directory where to save the model and the decoder.")
    parser.add_argument("--text_model", type=str, help="Repository id or path to the text encoder.")
    parser.add_argument("--audio_model", type=str, help="Repository id or path to the audio encoder.")
    parser.add_argument("--prompt_tokenizer", type=str, help="Repository id or path to the prompt tokenizer.")
    parser.add_argument("--num_codebooks", default=8, type=int, help="Number of codebooks to use in the model.")

    args = parser.parse_args()

    text_model = args.text_model
    encodec_version = args.audio_model
    prompt_tokenizer = args.prompt_tokenizer

    t5 = AutoConfig.from_pretrained(text_model)
    encodec = AutoConfig.from_pretrained(encodec_version)
    tokenizer = AutoTokenizer.from_pretrained(prompt_tokenizer)

    encodec_vocab_size = encodec.codebook_size
    num_codebooks = args.num_codebooks
    print("num_codebooks", num_codebooks)
    print("vocab size tokenizer", len(tokenizer))

    decoder_config = PrefixLMParlerTTSDecoderConfig(
        vocab_size=encodec_vocab_size + 1,
        max_position_embeddings=5120,  # 30 s = 2580
        num_hidden_layers=32,
        ffn_dim=4096,
        num_attention_heads=16,
        num_key_value_heads=8,
        use_cache=True,
        activation_function="silu",
        hidden_size=1024,
        attention_dropout=0.0,
        pad_token_id=encodec_vocab_size,
        eos_token_id=encodec_vocab_size,
        bos_token_id=encodec_vocab_size + 1,
        num_codebooks=num_codebooks,
        rope_theta=100_000,
        use_fused_lm_heads=True
    )

    decoder = PrefixLMParlerTTSForCausalLM(decoder_config)
    decoder.save_pretrained(os.path.join(args.save_directory, "decoder"))

    model = PrefixLMParlerTTSForConditionalGeneration.from_sub_models_pretrained(
        text_encoder_pretrained_model_name_or_path=text_model,
        audio_encoder_pretrained_model_name_or_path=encodec_version,
        decoder_pretrained_model_name_or_path=os.path.join(args.save_directory, "decoder"),
        vocab_size=len(tokenizer),
    )

    # set the appropriate bos/pad token ids
    model.generation_config.decoder_start_token_id = encodec_vocab_size + 1
    model.generation_config.pad_token_id = encodec_vocab_size
    model.generation_config.eos_token_id = encodec_vocab_size
    model.generation_config.bos_token_id = encodec_vocab_size + 1

    # set other default generation config params
    model.generation_config.max_new_tokens = int(30 * model.audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True


    model.config.pad_token_id = encodec_vocab_size
    model.config.decoder_start_token_id = encodec_vocab_size + 1

    model.save_pretrained(os.path.join(args.save_directory, "parler-tts-untrained-prefixlm/"))
