"""Sentence Transformer Finetuning Engine."""

from typing import Any, Optional

from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.finetuning.embeddings.common import (
    EmbeddingQAFinetuneDataset,
)
from llama_index.finetuning.types import BaseEmbeddingFinetuneEngine


from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import TrainingArguments, Trainer
from transformers.optimization import get_linear_schedule_with_warmup
import torch
import os
from typing import Optional, Any

class SentenceTransformersFinetuneEngine(BaseEmbeddingFinetuneEngine):
    """Sentence Transformers Finetune Engine."""

    def __init__(
        self,
        dataset: EmbeddingQAFinetuneDataset,
        model_id: str = "BAAI/bge-small-en",
        model_output_path: str = "exp_finetune",
        batch_size: int = 10,
        val_dataset: Optional[EmbeddingQAFinetuneDataset] = None,
        loss: Optional[Any] = None,
        epochs: int = 2,
        show_progress_bar: bool = True,
        evaluation_steps: int = 50,
    ) -> None:
        """Init params."""
        self.dataset = dataset
        self.model_id = model_id
        self.model_output_path = model_output_path

        self.model = SentenceTransformer(model_id)
        self.model = self.model.to('cuda:0')

        # Wrap the model with DataParallel to utilize multiple GPUs
        num_gpus = torch.cuda.device_count()
        devices = [f'cuda:{i}' for i in range(num_gpus)]
        self.model = torch.nn.DataParallel(self.model, device_ids=[device.idx for device in devices])

        self.examples = [InputExample(texts=[query, dataset.corpus[dataset.relevant_docs[query_id][0]]])
                         for query_id, query in dataset.queries.items()]

        self.loader = DataLoader(self.examples, batch_size=batch_size)

        self.evaluator = None
        if val_dataset is not None:
            self.evaluator = InformationRetrievalEvaluator(
                val_dataset.queries, val_dataset.corpus, val_dataset.relevant_docs
            )

        self.loss = loss or losses.MultipleNegativesRankingLoss(self.model)
        self.epochs = epochs
        self.show_progress_bar = show_progress_bar
        self.evaluation_steps = evaluation_steps
        self.warmup_steps = int(len(self.loader) * epochs * 0.1)

    def finetune(self, **train_kwargs: Any) -> None:
        """Finetune model."""
        output_dir = self.model_output_path
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=train_kwargs.get('per_device_train_batch_size', 10),
            save_steps=train_kwargs.get('save_steps', 100),
            save_total_limit=train_kwargs.get('save_total_limit', 2),
            evaluation_strategy=train_kwargs.get('evaluation_strategy', 'steps'),
            eval_steps=self.evaluation_steps,
            logging_dir=train_kwargs.get('logging_dir', os.path.join(output_dir, 'logs')),
            logging_steps=train_kwargs.get('logging_steps', 100),
            learning_rate=train_kwargs.get('learning_rate', 2e-5),
            warmup_steps=self.warmup_steps,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.loader,
            data_collator=train_kwargs.get('data_collator', None),
            compute_metrics=train_kwargs.get('compute_metrics', None),
            callbacks=train_kwargs.get('callbacks', None),
            tb_writer=train_kwargs.get('tb_writer', None),
            optimizers=train_kwargs.get('optimizers', None),
            scheduler=train_kwargs.get('scheduler', get_linear_schedule_with_warmup),
        )

        trainer.train()


    def get_finetuned_model(self, **model_kwargs: Any) -> BaseEmbedding:
        """Gets finetuned model."""
        embed_model_str = "local:" + self.model_output_path
        return resolve_embed_model(embed_model_str)
