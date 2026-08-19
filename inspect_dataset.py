from datasets import load_dataset

dataset = load_dataset("AI-Lab-Makerere/beans")

print(dataset)
print()
print("Features:")
print(dataset["train"].features)

example = dataset["train"][0]

print()
print("First example:")
print("Label ID:", example["labels"])

label_names = dataset["train"].features["labels"].names
print("Label name:", label_names[example["labels"]])

example["image"].save("sample_bean_image.jpg")
print("Saved sample_bean_image.jpg")
