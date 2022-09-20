from typing import List

BASE_CATEGORIES = ["Astrophisics", "Audio", "Sequential","Graphs", "Reinforcement Learning", "Natural Language Processing", "Computer Vision"]

def getCategories(base_categories: List[str], all_categories: List[str], additional_categories: List[str]) -> List[str]:
	"""
	Considering base_categories and all the category inputs, returns the final category list.

	Params
	-----------
	base_categories: (List[str]) List of base categories.
	all_categories: (List[str]) List of all categories. If evaluated to True, this list will be returned.
	additional_categories: (List[str]) List of additional categories. Added to base_categories if given and all_categories is evaluated to False.
	
	Return
	-----------
	(List[str]) List of final categories.
	"""
	if all_categories:
		return all_categories.copy()
	elif additional_categories:
		return base_categories.copy() + additional_categories.copy()
	else:
		return base_categories.copy()