import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from transformers import RobertaTokenizer, RobertaModel, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import math
import logging

# ==================== 设置日志 ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 设置设备 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")

# ==================== 加载和预处理数据 ====================
# 加载数据 (使用Mozilla数据集为例)
data = pd.read_csv("../dataset/Mozilla_dataset/Mozilla_process_data_filtered_stricter.csv")
# data = pd.read_csv("../dataset/Eclipse_dataset/Eclipse_process_data_filtered_stricter.csv")
# data = pd.read_csv("../dataset/GCC_dataset/GCC_stricter.csv")

# 删除缺失值，包括 'processed_summary', 'processed_description', 和 'severity'
data = data.dropna(subset=['processed_summary', 'processed_description', 'severity'])

# 确保 'severity' 列不包含空字符串或其他无效值
data = data[data['severity'].astype(str).str.strip() != '']

# 分别获取摘要和描述
summary_texts = data['processed_summary'].tolist()
description_texts = data['processed_description'].tolist()
labels = data['severity'].tolist()

# 标签编码为整数
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 检查是否有未编码的标签（例如，空字符串等）
if len(label_encoder.classes_) != len(set(encoded_labels)):
    logger.warning("存在未编码的标签，检查数据预处理步骤。")

# 标签编码元数据
product_encoder = LabelEncoder()
component_encoder = LabelEncoder()
priority_encoder = LabelEncoder()

data['product_encoded'] = product_encoder.fit_transform(data['product'])
data['component_encoded'] = component_encoder.fit_transform(data['component'])
data['priority_encoded'] = priority_encoder.fit_transform(data['priority'])

# 获取元数据
product_data = data['product_encoded'].tolist()
component_data = data['component_encoded'].tolist()
priority_data = data['priority_encoded'].tolist()

# ==================== 划分数据集 ====================
# 将数据集划分为 80% 训练集和 20% 验证集
train_summary, val_summary, train_description, val_description, train_labels, val_labels, \
train_product, val_product, train_component, val_component, train_priority, val_priority = train_test_split(
    summary_texts, description_texts, encoded_labels, product_data, component_data, priority_data, 
    test_size=0.2,  # 80% 训练，20% 验证
    random_state=42,
    stratify=encoded_labels
)

logger.info(f"训练集大小: {len(train_labels)}")
logger.info(f"验证集大小: {len(val_labels)}")

# ==================== 加载RoBERTa分词器 ====================
tokenizer = RobertaTokenizer.from_pretrained("../roberta-base")

# 数据编码（摘要与描述分别编码）
def encode_texts(texts, tokenizer, max_length=256):
    return tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

train_summary_encodings = encode_texts(train_summary, tokenizer)
train_description_encodings = encode_texts(train_description, tokenizer)
val_summary_encodings = encode_texts(val_summary, tokenizer)
val_description_encodings = encode_texts(val_description, tokenizer)

# 自定义Dataset类
class BugDataset(Dataset):
    def __init__(self, summary_encodings, description_encodings, labels, product, component, priority):
        self.summary_encodings = summary_encodings
        self.description_encodings = description_encodings
        self.labels = labels
        self.product = product
        self.component = component
        self.priority = priority

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'summary_input_ids': self.summary_encodings['input_ids'][idx],
            'summary_attention_mask': self.summary_encodings['attention_mask'][idx],
            'description_input_ids': self.description_encodings['input_ids'][idx],
            'description_attention_mask': self.description_encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'product': torch.tensor(self.product[idx], dtype=torch.long),
            'component': torch.tensor(self.component[idx], dtype=torch.long),
            'priority': torch.tensor(self.priority[idx], dtype=torch.long)
        }
        return item

# 创建Dataset实例
train_dataset = BugDataset(train_summary_encodings, train_description_encodings, train_labels, train_product,
                           train_component, train_priority)
val_dataset = BugDataset(val_summary_encodings, val_description_encodings, val_labels, val_product, val_component,
                         val_priority)

# ==================== 定义增强复杂性的多输入模型并使用Attention机制融合 ====================
class MultiInputModelWithAttention(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super(MultiInputModelWithAttention, self).__init__()
        self.roberta = RobertaModel.from_pretrained("../roberta-base")
        
        # 完全解冻RoBERTa模型的所有参数
        for param in self.roberta.parameters():
            param.requires_grad = True

        # 元数据嵌入
        self.fc_product = nn.Embedding(len(product_encoder.classes_), 64)
        self.fc_component = nn.Embedding(len(component_encoder.classes_), 64)
        self.fc_priority = nn.Embedding(len(priority_encoder.classes_), 64)

        # 获取hidden_size
        hidden_size = self.roberta.config.hidden_size

        # 注意力机制：从描述的last_hidden_state中选择与摘要相关的信息
        # 使用缩放点积注意力(Scaled Dot-Product Attention)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.sqrt_d = math.sqrt(hidden_size)

        # 融合后的维度：hidden_size(摘要) + hidden_size(描述Attention输出) + 32*3元数据
        fusion_dim = hidden_size + hidden_size + 64 * 3
        
        # 增加分类器的复杂性：MLP + LayerNorm + Dropout
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),  # 增加第一层的宽度
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        

        # 定义损失函数，可以根据需要选择是否使用类别权重
        if class_weights is not None:
            self.loss_fn = CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = CrossEntropyLoss()

    def forward(self, summary_input_ids, summary_attention_mask,
                description_input_ids, description_attention_mask,
                product, component, priority, labels=None):

        # 编码摘要
        summary_outputs = self.roberta(input_ids=summary_input_ids, attention_mask=summary_attention_mask)
        # pooler_output作为摘要全局表示: [batch_size, hidden_size]
        summary_embedding = summary_outputs.pooler_output

        # 编码描述
        description_outputs = self.roberta(input_ids=description_input_ids, attention_mask=description_attention_mask)
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        description_hidden = description_outputs.last_hidden_state

        # Attention: 使用summary_embedding作为query，从description_hidden中提取信息
        query = self.query_proj(summary_embedding).unsqueeze(1)  # [batch_size, 1, hidden_size]
        key = self.key_proj(description_hidden)  # [batch_size, seq_len, hidden_size]
        value = self.value_proj(description_hidden)  # [batch_size, seq_len, hidden_size]

        # 计算注意力分数: [batch_size, 1, seq_len]
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.sqrt_d
        # 对attention mask进行处理，将无效位置设为-Inf
        # 因为description_attention_mask: [batch_size, seq_len],需扩展维度与attention_scores匹配
        description_attention_mask_expanded = description_attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        # 将mask为0的地方设为非常低的分数
        attention_scores = attention_scores.masked_fill(description_attention_mask_expanded == 0, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, 1, seq_len]

        # 加权求和得到描述的聚合表示: [batch_size, hidden_size]
        context = torch.matmul(attention_weights, value).squeeze(1)  # [batch_size, hidden_size]

        # 元数据编码
        product_embedding = self.fc_product(product)  # [batch_size, 32]
        component_embedding = self.fc_component(component)  # [batch_size, 32]
        priority_embedding = self.fc_priority(priority)  # [batch_size, 32]

        # 融合：summary_embedding + context(描述Attention输出) + 元数据嵌入拼接
        fused_embedding = torch.cat(
            [summary_embedding, context, product_embedding, component_embedding, priority_embedding], dim=1)

        logits = self.classifier(fused_embedding)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        else:
            return logits

# ==================== 设置类别权重 ====================
# 手动设置类别权重
# 假设类别顺序为 [0: 'blocker', 1: 'critical', 2: 'major', 3: 'minor', 4: 'trivial']
# 您可以根据需要调整这些权重
custom_class_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
class_weights = torch.tensor(custom_class_weights, dtype=torch.float).to(device)
logger.info(f"手动设置的类别权重: {class_weights}")

# ==================== 初始化模型 ====================
num_classes = len(label_encoder.classes_)
model = MultiInputModelWithAttention(num_classes=num_classes, class_weights=class_weights)
model.to(device)
logger.info("模型已初始化并移动到设备上。")

# ==================== 定义训练参数 ====================
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",            # 改为每隔一定步数评估一次
    eval_steps=100,                  # 设置评估步数为1000
    save_strategy="steps",            # 改为每隔一定步数保存模型
    save_steps=100,                  # 设置保存步数为1000，与评估步数对齐
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    logging_dir="./logs",
    save_total_limit=3,
)

# ==================== 定义评估指标 ====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        "accuracy": report['accuracy'],
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1": report['weighted avg']['f1-score'],
    }

# ==================== 自定义 collate_fn ====================
def collate_fn(batch):
    summary_input_ids = torch.stack([item['summary_input_ids'] for item in batch])
    summary_attention_mask = torch.stack([item['summary_attention_mask'] for item in batch])
    description_input_ids = torch.stack([item['description_input_ids'] for item in batch])
    description_attention_mask = torch.stack([item['description_attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    product = torch.stack([item['product'] for item in batch])
    component = torch.stack([item['component'] for item in batch])
    priority = torch.stack([item['priority'] for item in batch])

    return {
        "summary_input_ids": summary_input_ids,
        "summary_attention_mask": summary_attention_mask,
        "description_input_ids": description_input_ids,
        "description_attention_mask": description_attention_mask,
        "product": product,
        "component": component,
        "priority": priority,
        "labels": labels
    }

# ==================== 初始化 Trainer ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # 使用验证集进行评估
    # 移除 tokenizer 参数以消除 FutureWarning
    # tokenizer=tokenizer,  # 如果您仍需要tokenizer，可以保留，但可能会有警告
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    data_collator=collate_fn  # 使用自定义collate_fn
)

# ==================== 验证RoBERTa模型是否完全解冻 ====================
def check_frozen_layers(model):
    frozen_layers = []
    for name, param in model.roberta.named_parameters():
        if not param.requires_grad:
            frozen_layers.append(name)
    if len(frozen_layers) == 0:
        print("所有RoBERTa层都已解冻。")
    else:
        print(f"以下RoBERTa层被冻结了: {frozen_layers}")

check_frozen_layers(model)

# ==================== 开始训练 ====================
trainer.train()

# ==================== 验证 ====================
evaluation = trainer.evaluate()
print("Validation Evaluation Results:", evaluation)

# ==================== 预测并生成分类报告（验证集） ====================
val_predictions = trainer.predict(val_dataset)
val_predicted_labels = torch.argmax(torch.tensor(val_predictions.predictions), dim=-1).numpy()

print("Classification Report on Validation Set:")
# 获取分类报告为字典
report_dict = classification_report(
    val_labels, val_predicted_labels,
    target_names=label_encoder.classes_,
    zero_division=0,
    output_dict=True
)

# 转换为 DataFrame 并四舍五入到四位小数
report_df = pd.DataFrame(report_dict).transpose().round(4)

print(report_df)


# ==================== 计算并输出指标（验证集） ====================
val_accuracy = accuracy_score(val_labels, val_predicted_labels)
val_precision = precision_score(val_labels, val_predicted_labels, average='weighted', zero_division=0)
val_recall = recall_score(val_labels, val_predicted_labels, average='weighted', zero_division=0)
val_f1 = f1_score(val_labels, val_predicted_labels, average='weighted', zero_division=0)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")

# ==================== 预测并生成分类报告（训练集） ====================
# train_predictions = trainer.predict(train_dataset)
# train_predicted_labels = torch.argmax(torch.tensor(train_predictions.predictions), dim=-1).numpy()

# print("Classification Report on Training Set:")
# print(classification_report(train_labels, train_predicted_labels, target_names=label_encoder.classes_))

# # ==================== 计算并输出指标（训练集） ====================
# train_accuracy = accuracy_score(train_labels, train_predicted_labels)
# train_precision = precision_score(train_labels, train_predicted_labels, average='weighted', zero_division=0)
# train_recall = recall_score(train_labels, train_predicted_labels, average='weighted', zero_division=0)
# train_f1 = f1_score(train_labels, train_predicted_labels, average='weighted', zero_division=0)

# print(f"Training Accuracy: {train_accuracy:.4f}")
# print(f"Training Precision: {train_precision:.4f}")
# print(f"Training Recall: {train_recall:.4f}")
# print(f"Training F1 Score: {train_f1:.4f}")
