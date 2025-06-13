# 🧠 BrainCoder

**BrainCoder** is a research project focused on decoding visual stimuli from EEG (Electroencephalogram) signals.  
The goal is to **learn generalized features** from brain activity that can accurately predict image representations across different subjects, enabling cross-subject decoding.

This project explores how to align EEG signals with powerful visual-semantic embeddings like [CLIP](https://openai.com/research/clip).

---

## 📚 Project Overview

- **Problem:** EEG signals are noisy and highly subject-specific, making cross-subject generalization challenging.
- **Solution:** Train models that map EEG signals to the corresponding **CLIP image features** while minimizing subject-specific variance.
- **Key idea:**  
  - Learn **subject-invariant** EEG representations.
  - Align EEG encodings with **vision-language embeddings**.
  
---

## 🛠️ Code Structure

```
BrainCoder/
├── configs/         # YAML config files for experiments
├── dataset/         # EEG Dataset processing and loading
├── archs/           # model archs to use
├── metrics/         # Evaluation metrics
├── models/          # EEG encoders, CLIP-based models
├── utils/           # Utility functions (logging, checkpointing)
├── train.py         # Entry point for training
├── eval.py          # Entry point for evaluation
├── requirements.txt # Required Python packages
└── README.md        # (You're reading it!)
```


---

## 🧠 Dataset

- Things EEG2 dataset
- EEG recordings collected while subjects viewed images.
- Each EEG signal is paired with the corresponding **CLIP image embedding**.


---

## ✨ Features

- Flexible EEG encoder architectures.
- Subject-invariant training (domain adversarial techniques planned).
- Contrastive learning between EEG features and image embeddings.
- Modular and easily extensible codebase.

---

## 📈 TODOs
- [ ] Update readme.md
- [ ] Add domain adversarial training to improve cross-subject generalization.
- [ ] Extend to zero-shot classification using CLIP text embeddings.
- [ ] Improve visualization of EEG feature space (e.g., t-SNE plots).
- [ ] Add pretrained model checkpoints.
- [ ] Add DinoV2 Embeddings.

---

## 🤝 Contributing

Contributions, ideas, and discussions are welcome!  
Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP) for image embeddings.
- [NICE-EEG](https://github.com/eeyhsong/NICE-EEG) for base code and dataset setup.