import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import List, Dict, Tuple
from collections import Counter
from PIL import Image
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer

def plot_class_distribution(labels: List[int], save_path: str = None):
    """Plot class distribution histogram."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=labels)
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_wordcloud(texts: List[str], save_path: str = None):
    """Create and plot wordcloud from text data."""
    # Calculate TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    
    # Get word frequencies weighted by TF-IDF
    word_scores = {}
    feature_names = tfidf.get_feature_names_out()
    
    for i in range(len(texts)):
        for j in range(len(feature_names)):
            score = tfidf_matrix[i, j]
            if score > 0:
                word_scores[feature_names[j]] = word_scores.get(
                    feature_names[j], 0
                ) + score

    # Create and generate a word cloud image
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(word_scores)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_sample_memes(
    images: List[Image.Image],
    texts: List[str],
    labels: List[int],
    num_samples: int = 5,
    save_path: str = None
):
    """Plot sample memes with their text and labels."""
    fig, axes = plt.subplots(
        1, num_samples,
        figsize=(4*num_samples, 4)
    )
    
    if num_samples == 1:
        axes = [axes]
    
    for i, (img, text, label) in enumerate(
        zip(images[:num_samples], texts[:num_samples], labels[:num_samples])
    ):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(
            f"Label: {label}\n{text[:50]}{'...' if len(text) > 50 else ''}",
            wrap=True
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: str = None
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(
    y_true: List[int],
    y_prob: List[float],
    save_path: str = None
):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.2f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str],
    save_path: str = None
):
    """Plot training history for specified metrics."""
    plt.figure(figsize=(12, 4))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, len(metrics), i)
        for split in ['train', 'val']:
            key = f'{split}_{metric}'
            if key in history:
                plt.plot(
                    history[key],
                    label=split.capitalize()
                )
        plt.title(f'{metric.capitalize()} History')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 