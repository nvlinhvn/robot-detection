
import re
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import urlparse, unquote
from sklearn.metrics import log_loss, roc_curve, auc, precision_recall_curve
from scipy.stats import ks_2samp
import scikitplot as skplt
from nltk.corpus import stopwords
import spacy

# Assuming you're using the Dutch language model for spaCy
nlp = spacy.load('nl_core_news_sm')

def process_dutch_text(input_text):
    """
    Process Dutch text by removing digits, lowercasing, and filtering stopwords.
    Simply cleaning text from product content, remove stop words, to be prepared for topic modeling
    Args:
        input_text (str): The input text to process.

    Returns:
        str: The processed text, or None if an error occurs.
    """
    try:
        # Remove words containing digits
        input_text = re.sub(r'\b\w*\d\w*\b', '', input_text)

        # Split the text into words and remove hyphens
        words = input_text.split('-')

        # Lowercase each word
        processed_words = [word.lower() for word in words]

        # Part-of-speech tagging using spaCy
        doc = nlp(' '.join(processed_words))

        # Filter out tokens that consist solely of digits
        processed_words = [token.text for token in doc if not token.text.isdigit()]

        # Remove stopwords
        stop_words = set(stopwords.words('dutch'))

        # Filter out specific terms
        processed_words = [
            word for word in processed_words
            if word not in stop_words
            and len(word) > 2
            and word not in ('euro', 'cent', 'meter', 'meters', 'html')
        ]

        # Join the words into a single string
        processed_text = ' '.join(processed_words)

        return processed_text
    except Exception as e:
        print(f"Error processing text '{input_text}': {e}")
        return None

def extract_product_content(url):
    """
    Extract the product content from a URL.
    The assumption is that Robot might go only to some specific products

    Args:
        url (str): The URL to extract content from.

    Returns:
        str: The extracted product content or "Others" if an error occurs.
    """
    try:
        # Parse the URL
        parsed_url = urlparse(url)

        # Extract the last path component and unquote it
        last_path_component = unquote(parsed_url.path.split('/')[3])

        return last_path_component
    except Exception:
        return "Others"

def metrics(y_test, y_pred_prob, model_name=None):
    """
    Calculate and plot various metrics for binary classification.
    Plot and compare the prediction vs. truth values in different metrics
    - Log-loss </p>
    - Precision-Recall </p>
    - Receiver Operating Characteristic </p>
    - KS-score
    - Lift 
    - Calibration
    - Cumulative Gains Curve
    Args:
        y_test (array-like): True binary labels.
        y_pred_prob (array-like): Predicted probabilities for the positive class.

    Returns:
        None. Displays a plot with various metrics.
    """
    ks_y_pred_prob = y_pred_prob
    y_pred_prob = y_pred_prob[:, 1]

    class_weights = {0: 1, 1: 0.6}  # Fix class weights
    sample_weight = np.where(y_test == 1, class_weights[1], class_weights[0])

    # Log-loss metrics
    weighted_log_loss = log_loss(y_test, y_pred_prob, sample_weight=sample_weight)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    auc_pr = auc(recall, precision)

    # Calculate KS statistic
    ks_statistic, p_value = ks_2samp(y_pred_prob[y_test == 1], y_pred_prob[y_test == 0])

    # Plot metrics
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].hist(y_pred_prob[y_test == 1], bins=100, color="blue", edgecolor="black", label="class 1")
    axs[0, 0].hist(y_pred_prob[y_test == 0], bins=100, color="green", alpha=0.5, edgecolor="black", label="class 0")
    axs[0, 0].set_xlabel("Probability", color="white", fontsize=14)
    axs[0, 0].set_ylabel("Frequency", color="white", fontsize=14)
    axs[0, 0].set_title(f"Histogram of Predicted Probabilities\nWeighted Log Loss = {round(weighted_log_loss, 2)}", color="white")
    axs[0, 0].tick_params(axis="both", colors="white")
    axs[0, 0].legend()

    axs[0, 1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    axs[0, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axs[0, 1].set_xlim([0.0, 1.0])
    axs[0, 1].set_ylim([0.0, 1.05])
    axs[0, 1].set_xlabel("False Positive Rate", color="white", fontsize=14)
    axs[0, 1].set_ylabel("True Positive Rate", color="white", fontsize=14)
    axs[0, 1].set_title("Receiver Operating Characteristic", color="white")
    axs[0, 1].legend(loc="lower right")
    axs[0, 1].tick_params(axis="both", colors="white")

    axs[1, 0].plot(recall, precision, color="darkorange", lw=2, label=f"AUC = {auc_pr:.2f}")
    axs[1, 0].set_xlim([0.0, 1.0])
    axs[1, 0].set_ylim([0.0, 1.05])
    axs[1, 0].set_xlabel("Recall", color="white", fontsize=14)
    axs[1, 0].set_ylabel("Precision", color="white", fontsize=14)
    axs[1, 0].set_title("Precision-Recall Curve", color="white")
    axs[1, 0].legend(loc="lower left")
    axs[1, 0].tick_params(axis="both", colors="white")

    # Plot the K-S plot
    skplt.metrics.plot_ks_statistic(y_test, ks_y_pred_prob, ax=axs[1, 1])
    
    axs[1, 1].set_xlabel("% of sample", color="white", fontsize=14)
    axs[1, 1].set_ylabel("CDF(x)", color="white", fontsize=14)
    axs[1, 1].set_title(f"KS CDF curve normalize: {round(ks_statistic, 2)} \n p-value = {p_value}", color="white")
    axs[1, 1].legend()
    axs[1, 1].tick_params(axis="both", colors="white")

    plt.tight_layout()
    plt.show()
    
    # Additional plots
    # Plot 5: Cumulative Gains Curve
    skplt.metrics.plot_cumulative_gain(y_test, ks_y_pred_prob)
    plt.tight_layout()
    plt.show()

    # Plot 6: Lift Curve
    skplt.metrics.plot_lift_curve(y_test, ks_y_pred_prob)
    plt.tight_layout()
    plt.show()

    # Plot 7: Calibration Curve
    skplt.metrics.plot_calibration_curve(y_test, [ks_y_pred_prob], ['Model'])
    plt.tight_layout()
    plt.show()
