# Step 1 — Character Classifier (CNN)

First milestone of the project: given a 32x32 px image crop containing
**a single character**, predict which one it is (62 classes: `0-9`, `A-Z`, `a-z`).

This isn't "reading text" yet — it's the base building block everything
else is built on. Before trying to read a full word, we need to nail the
simpler problem first: recognizing an isolated character.

## Why synthetic text instead of a dataset like MNIST

MNIST/EMNIST are **handwritten** digits and letters, with a lot of variation
between people. Our real goal is to read **digital/printed** text
(screenshots, documents, code), where the shape of each letter is much more
consistent because it depends on the font, not on someone's handwriting.
That's why we generate our own dataset by rendering text with different
system fonts — this way we control the variety (font, size, slight
rotation, noise) and can generate as many examples as we want without
depending on downloading anything external.

## Deep learning concepts covered here

- **CNN (convolutional neural network)**: why convolutions instead of
  dense layers directly on pixels — convolutions detect local patterns
  (edges, curves) regardless of where in the image they appear.
- **BatchNorm**: normalizes activations between layers, helps training be
  faster and more stable.
- **MaxPooling**: progressively reduces spatial resolution, keeping the
  most relevant information from each region.
- **Dropout**: randomly turns off neurons during training to prevent the
  network from memorizing instead of generalizing (overfitting).
- **CrossEntropyLoss**: the standard loss function for multiclass
  classification — compares the predicted probability distribution against
  the correct class.
- **Data augmentation**: applying random transformations (rotation,
  translation, scale) each epoch so the model sees variations instead of
  memorizing exact images.
- **train/val split**: why we measure progress on a set the model NEVER
  trains on, to detect overfitting.

## How to run it

```bash
# 1. Generate the synthetic dataset (from data_generator/)
cd ../data_generator
python generate_chars.py --out ../data/chars --per_class 500

# 2. Train (from 01_char_classifier/)
cd ../01_char_classifier
python train.py --data ../data/chars --epochs 15

# 3. Test the model on a single image
python predict.py --checkpoint checkpoints/best_model.pt --image my_letter.png
```

## Reference results

With 500 images per class (31,000 total images) and 15 epochs: around
**94-95% validation accuracy** already within the first 5 epochs, training
on CPU. More data, more epochs, or more aggressive data augmentation should
push this higher.