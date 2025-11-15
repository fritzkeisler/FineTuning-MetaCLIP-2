# Fine-Tuning MetaCLIP-2 for Image Classification on Downstream Tasks
 

![0](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/2ahO3T_m4xNqgPbR0y5rt.png)

## [1.] Introduction

MetaCLIP2 is Meta's breakthrough recipe for scaling contrastive language–image pretraining (CLIP) worldwide, advancing beyond English-only models to support 300+ languages and diverse global contexts. Unlike previous approaches, MetaCLIP2 is trained from scratch on native multilingual image–text pairs collected without relying on private data, machine translation, or distillation; it introduces innovations in scalable metadata construction, per-language data curation algorithms, and a refined training framework to overcome the "curse of multilinguality," where model performance drops as languages diversify. 

The MetaCLIP2 recipe fundamentally improves multilingual benchmarks, setting new state-of-the-art results for zero-shot classification, retrieval, and region-specific recognition tasks such as CVQA, Babel‑ImageNet, and XM3600, outperforming previous systems like mSigLIP and SigLIP‑2. Notably, it demonstrates that English and non-English data can mutually benefit each other, enabling the model to better generalize and remain robust in both English and multilingual scenarios. MetaCLIP2's open “no-filter philosophy” ensures inclusive training with genuinely global data, minimizing dataset and pipeline bias. 

MetaCLIP2 is highly efficient in zero-shot image classification, achieving about 81.3% top-1 accuracy on ImageNet with the ViT-H/14 model, outperforming its English-only counterpart by 0.8% and surpassing other multilingual models like mSigLIP by 0.7%. Its strength stems from training on large-scale worldwide web image-text pairs, which enables robust generalization and state-of-the-art performance across both English and multilingual benchmarks without special architectural changes. However, MetaCLIP2 may require downstream finetuning when applied to domain-specific tasks, specialized datasets, or vision-language applications that differ significantly from its training data, such as medical imaging, satellite imagery, or custom classification labels, to improve accuracy and adaptation. Overall, it excels out-of-the-box for general zero-shot classification but benefits from finetuning in tailored, niche contexts.

![1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/A7HIDAF_ddf44S0DeccGQ.png)

<p align="center">
Image (a): Meta CLIP 2: A Worldwide Scaling Recipe <strong>[Paper]</strong> <strong>[Page 4]</strong><br>
https://arxiv.org/pdf/2507.22062
</p>


The model architecture closely follows the original CLIP, featuring joint image and text encoders optimized for worldwide scaling, and is fully supported in frameworks such as Hugging Face Transformers for further research and application deployment. The following topics discuss the step-by-step process for fine-tuning MetaCLIP-2. To illustrate this, we will use the CIFAR-10 dataset, which contains labeled images for classification. 

## [2.] Step by Step Process for Fine Tuning MetaCLIP 2 for Image Classification

The following steps demonstrate how to fine tune [MetaCLIP 2](https://huggingface.co/facebook/metaclip-2-worldwide-s16), a powerful multilingual vision and language encoder, for downstream image classification tasks. While MetaCLIP 2 achieves strong zero shot performance across multilingual benchmarks, fine tuning enables the model to specialize in targeted downstream domains where task specific accuracy is important. Through supervised learning on datasets like CIFAR 10, the model shifts from broad semantic alignment to optimized feature learning for class level prediction. This process strengthens decision boundaries, improves label mapping reliability, and enhances generalization for real world deployment in domain driven image classification tasks.

### [i] Install the packages

```py
%%capture
!pip install evaluate datasets accelerate
!pip install transformers torchvision
!pip install huggingface-hub hf_xet
#Hold tight, this will take around 1-2 minutes.
```

**Dataset ID2Label Mapping**

Note : The `id2label` mapping shows how numerical class IDs correspond to human-readable labels.  
This is **not required** for training or evaluation — it's just for **visual reference** and **debugging**.

    To demonstrate the fine-tuning process, we will use the CIFAR-10 dataset, which contains labeled images for image classification.
    You can find the CIFAR-10 dataset here: [cifar10](https://huggingface.co/datasets/uoft-cs/cifar10)


```py
from datasets import load_dataset
dataset = load_dataset("uoft-cs/cifar10")
labels = dataset["train"].features["label"].names
id2label = {str(i): label for i, label in enumerate(labels)}
print(id2label)
```

> Loading Dataset ...

```
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:104: UserWarning: 
Error while fetching `HF_TOKEN` secret value from your vault: 'Requesting secret HF_TOKEN timed out. Secrets can only be fetched when running from the Colab UI.'.
You are not authenticated with the Hugging Face Hub in this notebook.
If the error persists, please let us know by opening an issue on GitHub (https://github.com/huggingface/huggingface_hub/issues/new).
  warnings.warn(
README.md:  5.16k/? [00:00<00:00, 143kB/s]plain_text/train-00000-of-00001.parquet: 100% 120M/120M [00:01<00:00, 110MB/s]plain_text/test-00000-of-00001.parquet: 100% 23.9M/23.9M [00:00<00:00, 48.3MB/s]Generating train split: 100% 50000/50000 [00:02<00:00, 69843.92 examples/s]Generating test split: 100% 10000/10000 [00:02<00:00, 745.32 examples/s]

{'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat',
'4': 'deer', '5': 'dog', '6': 'frog', '7': 'horse', '8': 'ship',
'9': 'truck'}
```

### [ii] Import modules required for data manipulation, model training, and image preprocessing.

```py
import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

from transformers import AutoImageProcessor, AutoProcessor
from transformers.image_utils import load_image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)

from PIL import Image, ExifTags
from PIL import Image as PILImage
from PIL import ImageFile
# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

### [iii] Loading and Preparing the Dataset

```py
from datasets import load_dataset
dataset = load_dataset("uoft-cs/cifar10", split="train")

from pathlib import Path

file_names = []
labels = []

for example in dataset:
    file_path = str(example['img'])
    label = example['label']

    file_names.append(file_path)
    labels.append(label)

print(len(file_names), len(labels))
```

### [iv] Creating a DataFrame and Balancing the Dataset & Working with a Subset of Labels

> Manual Label List (for Custom Naming & Mapping Consistency)

We manually define the `labels_list` to:

    Avoid auto-mapping issues that may arise due to inconsistent label formats in the dataset.

    Support flexible naming conventions, especially when label names need to follow a specific format or order.

    Ensure consistent behavior across different tools (like `ClassLabel`, Hugging Face datasets, and visualization libraries).

```py
df = pd.DataFrame.from_dict({"img": file_names, "label": labels})
print(df.shape)

df.head()
df['label'].unique()

y = df[['label']]
df = df.drop(['label'], axis=1)
ros = RandomOverSampler(random_state=83)
df, y_resampled = ros.fit_resample(df, y)
del y
df['label'] = y_resampled
del y_resampled
gc.collect()

labels_subset = labels[:5]
print(labels_subset)

#labels_list = ['example_label_0', 'example_label_1'................,'example_label_n-1']
labels_list = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

label2id, id2label = {}, {}
for i, label in enumerate(labels_list):
    label2id[label] = i
    id2label[i] = label

ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

print("Mapping of IDs to Labels:", id2label, '\n')
print("Mapping of Labels to IDs:", label2id)
```

### [v] Mapping and Casting Labels

```py
def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example
```

### [vi] Splitting the Dataset

```py
dataset = dataset.map(map_label2id, batched=True)
dataset = dataset.cast_column('label', ClassLabels)
dataset = dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label")

train_data = dataset['train']
test_data = dataset['test']
```

### [vii] Setting Up the Model and Processor

```py
model_str = "facebook/metaclip-2-worldwide-s16"
processor = AutoImageProcessor.from_pretrained(model_str)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
```

### [viii] Defining Data Transformations

```py
_train_transforms = Compose([
    Resize((size, size)),
    RandomRotation(90),
    RandomAdjustSharpness(2),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])

_val_transforms = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])
```

### [ix] Applying Transformations to the Dataset

```py
def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

train_data.set_transform(train_transforms)
test_data.set_transform(val_transforms)
```

### [x] Creating a Data Collator

```py
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```

### [xi] Initializing the Model

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))
model.config.id2label = id2label
model.config.label2id = label2id

print(model.num_parameters(only_trainable=True) / 1e6)
```

### [xii] Defining Metrics and the Compute Function

```py
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']

    return {
        "accuracy": acc_score
    }
```

### [xiii] Setting Up Training Arguments

```py
args = TrainingArguments(
    output_dir="metaclip-2-image-classification/",
    logging_dir='./logs',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.02,
    warmup_steps=50,
    remove_unused_columns=False,
    save_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=4,
    report_to="none"
)
```

### [xiv] Initializing the Trainer

```py
trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)
```

### [xv] Evaluating, Training, and Predicting

```py
trainer.evaluate()

trainer.train()

trainer.evaluate()

outputs = trainer.predict(test_data)
print(outputs.metrics)
```

### [xvi] Computing Additional Metrics and Plotting the Confusion Matrix

```py
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):

    plt.figure(figsize=figsize)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.0f'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

if len(labels_list) <= 150:
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels_list, figsize=(8, 6))

print()
print("Classification report:")
print()
print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))
```

### [xvii] Saving the Model and Uploading to Hugging Face Hub

```py
trainer.save_model()
```

```py
from huggingface_hub import notebook_login, HfApi
notebook_login()
```

```py
api = HfApi()
repo_id = f"prithivMLmods/MetaCLIP-2-Cifar10"

api.upload_folder(
    folder_path="metaclip-2-image-classification/",
    path_in_repo=".",
    repo_id=repo_id,
    repo_type="model",
    revision="main"
)
```

| **Resource**                                | **Description**                                        | **Link** |
|---------------------------------------------|--------------------------------------------------------|---------|
| MetaCLIP 2 CIFAR10 Model                    | Fine tuned downstream CIFAR10 image classification model | https://huggingface.co/prithivMLmods/MetaCLIP-2-Cifar10 |
| MetaCLIP 2 Fine Tuning Notebook             | Quickstart notebook for MetaCLIP 2 finetuning            | https://github.com/PRITHIVSAKTHIUR/FineTuning-MetaCLIP-2/blob/main/meta_cllip2_finetune.ipynb |


## [3.] MetaCLIP 2 Zero Shot Image Classification Demo

> Try the MetaCLIP 2 Zero Shot Classification demo on Hugging Face Spaces: [https://huggingface.co/spaces/prithivMLmods/metaclip-2-demo](https://huggingface.co/spaces/prithivMLmods/metaclip-2-demo)


<iframe
	src="https://prithivmlmods-metaclip-2-demo.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

## [4.] Acknowledgements

| **Resource**                 | **Description**                                                | **Link** |
|--------------------------------------------|----------------------------------------------------------------|---------|
| Hugging Face Transformers                  | Library for training and inference with state of the art models | https://huggingface.co/docs/transformers |
| PyTorch                                    | Deep learning framework for GPU and CPU accelerated training     | https://pytorch.org |
| MetaCLIP 2 Paper                           | Worldwide scaling recipe for multilingual vision language models | https://huggingface.co/papers/2507.22062 |
| MetaCLIP 2 Models Collection               | Hub collection containing multilingual MetaCLIP 2 models         | https://huggingface.co/collections/merve/metaclip2-multilingual |
| Meta AI (facebook)                         | Research and development team that released MetaCLIP 2           | https://huggingface.co/facebook |
| MetaCLIP 2 Fine Tuning Notebook            | Example notebook for fine tuning MetaCLIP 2                     | https://github.com/PRITHIVSAKTHIUR/FineTuning-MetaCLIP-2 |

## [5.] Conclusion

In conclusion, fine tuning MetaCLIP 2 for image classification demonstrates the model’s adaptability and strong multilingual understanding even in domain specific downstream tasks. By leveraging its global vision and language alignment and well curated training recipe, MetaCLIP 2 efficiently learns discriminative visual features from datasets like CIFAR 10 while maintaining cross lingual robustness. The step by step process, from dataset preparation and augmentation to evaluation and deployment, shows that MetaCLIP 2 can serve as a universal image encoder capable of achieving high accuracy with limited fine tuning effort. This workflow establishes a strong foundation for extending MetaCLIP 2 to broader multilingual or domain adapted tasks across both academic and industrial applications.

Happy fine-tuning! 🤗
