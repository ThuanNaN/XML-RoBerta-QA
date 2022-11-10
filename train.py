from transformers import TrainingArguments
from models.mrc_model import MRCQuestionAnswering
from transformers import Trainer
from utils import data_loader
import argparse
import tqdm as tqdm
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_lst):
        self.dataset_lst = dataset_lst

    def __getitem__(self, idx):
        return self.dataset_lst[idx]

    def __len__(self):
        return len(self.dataset_lst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size training')
    parser.add_argument('--fp16', action='store_true', help='FP16 half-precision training')
    parser.add_argument('--fp16-opt-level', type=str, default="01", help='apex AMP optimization level selected in ["00", "01","02", "03" ]')
    opt = parser.parse_args()

    model = MRCQuestionAnswering.from_pretrained("xlm-roberta-base",
                                                 cache_dir='./model-bin/cache',
                                                 #local_files_only=True
                                                )
    print(model)
    print(model.config)

    train_dataset, valid_dataset = data_loader.get_dataloader(
        train_path='./data-bin/processed/train.dataset',
        valid_path='./data-bin/processed/valid.dataset'
    )

    data_set_train, data_set_val  = [] , []
    for data in train_dataset:
        if data["valid"] :
            data_set_train.append(data)

    for data in valid_dataset:
        if data["valid"]:
            data_set_val.append(data)

    dataset_train = CustomDataset(data_set_train)
    dataset_valid = CustomDataset(data_set_val)


    training_args = TrainingArguments("model-bin/test",
                                      do_train=True,
                                      do_eval=True,
                                      num_train_epochs=opt.epochs,
                                      learning_rate=1e-4,
                                      warmup_ratio=0.05,
                                      weight_decay=0.01,
                                      per_device_train_batch_size=opt.batch_size,
                                      per_device_eval_batch_size=opt.batch_size,
                                      fp16=opt.fp16,
                                      fp16_opt_level = opt.fp16_opt_level,
                                      gradient_accumulation_steps=1,
                                      logging_dir='./log',
                                      logging_steps=5,
                                      label_names=['start_positions',
                                                   'end_positions',
                                                   'span_answer_ids',
                                                   'input_ids',
                                                   'words_lengths'],
                                      group_by_length=True,
                                      save_strategy="epoch",
                                      metric_for_best_model='f1',
                                      load_best_model_at_end=True,
                                      save_total_limit=2,
                                      #eval_steps=1,
                                      #evaluation_strategy="steps",
                                      evaluation_strategy="epoch",
                                      )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        data_collator=data_loader.data_collator,
        compute_metrics=data_loader.compute_metrics
    )

    trainer.train(resume_from_checkpoint = False)
    trainer.save_model("./model-bin/checkpoints/last_checkpoint/")