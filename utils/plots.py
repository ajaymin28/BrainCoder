import matplotlib.pyplot as plt
import numpy as np


def plot_single_run_all_subjects(data):

    # Data for 100 epochs
    subjects = range(1, 11)
    top1_100 = [0.105, 0.080, 0.025, 0.045, 0.015, 0.055, 0.015, 0.020, 0.030, 0.030]
    top3_100 = [0.185, 0.140, 0.050, 0.105, 0.040, 0.095, 0.030, 0.035, 0.065, 0.075]
    top5_100 = [0.220, 0.165, 0.065, 0.120, 0.055, 0.110, 0.050, 0.075, 0.070, 0.100]

    # Calculate means
    mean_top1_100 = np.mean(top1_100)
    mean_top3_100 = np.mean(top3_100)
    mean_top5_100 = np.mean(top5_100)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25  # Width of bars
    x = np.arange(len(subjects))

    # Plot bars
    ax.bar(x - width, top1_100, width, label='Top1', color='blue')
    ax.bar(x, top3_100, width, label='Top3', color='green')
    ax.bar(x + width, top5_100, width, label='Top5', color='red')

    # Add mean lines
    ax.axhline(y=mean_top1_100, color='blue', linestyle='--', label=f'Top1 Mean: {mean_top1_100:.3f}')
    ax.axhline(y=mean_top3_100, color='green', linestyle='--', label=f'Top3 Mean: {mean_top3_100:.3f}')
    ax.axhline(y=mean_top5_100, color='red', linestyle='--', label=f'Top5 Mean: {mean_top5_100:.3f}')

    # Customize plot
    ax.set_title('Accuracy Metrics at 100 Epochs')
    ax.set_xlabel('Subject')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_all_subjects(data):

    # Data for 100 epochs
    subjects = range(1, 11)
    top1_100 = [0.105, 0.080, 0.025, 0.045, 0.015, 0.055, 0.015, 0.020, 0.030, 0.030]
    top3_100 = [0.185, 0.140, 0.050, 0.105, 0.040, 0.095, 0.030, 0.035, 0.065, 0.075]
    top5_100 = [0.220, 0.165, 0.065, 0.120, 0.055, 0.110, 0.050, 0.075, 0.070, 0.100]

    # Data for 200 epochs
    top1_200 = [0.060, 0.045, 0.030, 0.025, 0.020, 0.015, 0.030, 0.025, 0.025, 0.030]
    top3_200 = [0.125, 0.100, 0.045, 0.050, 0.035, 0.080, 0.045, 0.085, 0.040, 0.080]
    top5_200 = [0.145, 0.115, 0.050, 0.070, 0.045, 0.085, 0.060, 0.100, 0.055, 0.095]

    # Calculate means
    mean_top1_100 = np.mean(top1_100)
    mean_top3_100 = np.mean(top3_100)
    mean_top5_100 = np.mean(top5_100)
    mean_top1_200 = np.mean(top1_200)
    mean_top3_200 = np.mean(top3_200)
    mean_top5_200 = np.mean(top5_200)

    # Set up the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    width = 0.35
    x = np.arange(len(subjects))

    # Top1 comparison
    ax1.bar(x - width/2, top1_100, width, label='100 Epochs', color='blue')
    ax1.bar(x + width/2, top1_200, width, label='200 Epochs', color='orange')
    ax1.axhline(y=mean_top1_100, color='blue', linestyle='--', label=f'Mean 100: {mean_top1_100:.3f}')
    ax1.axhline(y=mean_top1_200, color='orange', linestyle='--', label=f'Mean 200: {mean_top1_200:.3f}')
    ax1.set_title('Top1 Accuracy')
    ax1.set_xlabel('Subject')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects)
    ax1.legend()

    # Top3 comparison
    ax2.bar(x - width/2, top3_100, width, label='100 Epochs', color='blue')
    ax2.bar(x + width/2, top3_200, width, label='200 Epochs', color='orange')
    ax2.axhline(y=mean_top3_100, color='blue', linestyle='--', label=f'Mean 100: {mean_top3_100:.3f}')
    ax2.axhline(y=mean_top3_200, color='orange', linestyle='--', label=f'Mean 200: {mean_top3_200:.3f}')
    ax2.set_title('Top3 Accuracy')
    ax2.set_xlabel('Subject')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects)
    ax2.legend()

    # Top5 comparison
    ax3.bar(x - width/2, top5_100, width, label='100 Epochs', color='blue')
    ax3.bar(x + width/2, top5_200, width, label='200 Epochs', color='orange')
    ax3.axhline(y=mean_top5_100, color='blue', linestyle='--', label=f'Mean 100: {mean_top5_100:.3f}')
    ax3.axhline(y=mean_top5_200, color='orange', linestyle='--', label=f'Mean 200: {mean_top5_200:.3f}')
    ax3.set_title('Top5 Accuracy')
    ax3.set_xlabel('Subject')
    ax3.set_ylabel('Accuracy')
    ax3.set_xticks(x)
    ax3.set_xticklabels(subjects)
    ax3.legend()

    plt.tight_layout()
    plt.show()