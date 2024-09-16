"""
Created on Sat Aug 17 19:22:43 2024
@author: Chenfeng Chen
"""

import torch
import evaluate
from datasets import load_dataset, Dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List
try:
    from google.colab import userdata
    hf_api_key = userdata.get('HF_TOKEN')
except ImportError:
    from config import keys
    hf_api_key = keys["hugging_face_api_key"]


class DatasetHandler:
    def __init__(self, dataset_name, language, subset_size):
        self.dataset_name = dataset_name
        self.language = language
        self.subset_size = subset_size
        self.subsets = ["train", "validation", "test"]

    def _load_subset(self, split):
        dataset_iter = load_dataset(self.dataset_name, self.language, split=split,
                                    token=hf_api_key, streaming=True, trust_remote_code=True)
        return list(dataset_iter.take(self.subset_size))

    def prepare_dataset(self):
        dataset = DatasetDict({
            split: Dataset.from_list(self._load_subset(split)) for split in self.subsets
        })
        dataset = dataset.select_columns(["audio", "sentence"])
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        return dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate input features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        labels = [{"input_ids": feature["labels"]} for feature in features]
        # Pad input features and labels
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")
        # Replace padding tokens with -100 to ignore in loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove the first token (BOS) if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class WhisperTrainer:
    def __init__(self, model_name, language, task, dataset_handler, use_pretrained=True):
        self.model_name = model_name
        self.language = language
        self.task = task
        self.dataset_handler = dataset_handler
        self.use_pretrained = use_pretrained
        self.processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.metric = evaluate.load("wer")
        self.model = self.initialize_model()

    def prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["input_features"] = self.feature_extractor(audio["array"],
                                                         sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch

    def process_common_voice(self):
        common_voice = self.dataset_handler.prepare_dataset()
        common_voice = common_voice.map(self.prepare_dataset, num_proc=4)
        return common_voice

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def initialize_model(self):
        if self.use_pretrained:
            # Load the pretrained model
            model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        else:
            # Initialize a new model with random parameters
            config = WhisperConfig.from_pretrained(self.model_name)
            model = WhisperForConditionalGeneration(config)
        # Set model configuration
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = None
        model.gradient_checkpointing_enable()
        model.config.max_length = 225
        return model

    def train(self, common_voice, output_dir="./whisper_finetuned"):
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_steps=10,
            max_steps=20,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=10,
            eval_steps=10,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=common_voice["train"],
            eval_dataset=common_voice["validation"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        trainer.train()
        trainer.save_model(output_dir)
        return trainer

    def evaluate(self, sample):
        input_features = self.processor.feature_extractor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt"
        ).input_features

        if torch.cuda.is_available():
            input_features = input_features.to("cuda")
            self.model.to("cuda")

        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(input_features)

        transcription = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Reference:", sample["sentence"])
        print("Transcription:", transcription)


# %% test
dataset_handler = DatasetHandler(dataset_name="mozilla-foundation/common_voice_16_1",
                                 language="zh-CN",
                                 subset_size=100)

# Set `use_pretrained` to False if you want to train from scratch
whisper_trainer = WhisperTrainer(model_name="openai/whisper-tiny",
                                 language="chinese",
                                 task="transcribe",
                                 dataset_handler=dataset_handler,
                                 use_pretrained=True)

common_voice = whisper_trainer.process_common_voice()
trainer = whisper_trainer.train(common_voice)

# Evaluate model
sample = common_voice["test"][0]
whisper_trainer.evaluate(sample)
