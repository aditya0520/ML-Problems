from collections import Counter
import numpy as np

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
	
	target_label_count = Counter(row[target_attr] for row in examples)
	
	if len(target_label_count) == 1:  
		return next(iter(target_label_count.keys()))
	if not attributes:  
		return max(target_label_count, key=target_label_count.get)

	
	total_elements = len(examples)
	base_entropy = -sum((count / total_elements) * np.log2(count / total_elements)
                        for count in target_label_count.values())
	
	def compute_info_gain(attribute):
		attribute_val_counter = Counter(row[attribute] for row in examples)
		total_entropy = 0
		for attr_val, attr_count in attribute_val_counter.items():
			subset = [row for row in examples if row[attribute] == attr_val]
			subset_label_count = Counter(row[target_attr] for row in subset)
			subset_entropy = -sum((count / len(subset)) * np.log2(count / len(subset))
                                  for count in subset_label_count.values())
			total_entropy += (attr_count / total_elements) * subset_entropy
		
		return base_entropy - total_entropy
	
	attribute_info_gain = {attr: compute_info_gain(attr) for attr in attributes}

	best_attribute = max(attribute_info_gain, key=attribute_info_gain.get)
	tree = {best_attribute: {}}

	attribute_values = set(row[best_attribute] for row in examples)

	for value in attribute_values:

		subset = [row for row in examples if row[best_attribute] == value]
		if not subset:
			majority_class = max(target_label_count, key=target_label_count.get)
			tree[best_attribute][value] = majority_class
		else:
			remaining_attributes = [attr for attr in attributes if attr != best_attribute]
			tree[best_attribute][value] = learn_decision_tree(subset, remaining_attributes, target_attr)
	return tree