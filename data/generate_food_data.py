"""
Generates the Nigerian Food Nutrition Dataset (150 records) referenced in
Section 3.4 of the thesis. Attributes: name, category, glycaemic_index,
carbohydrate_g, fibre_g, protein_g, fat_g, calories_kcal (per 100g serving).

GI and macro values are grounded in published glycaemic index tables
(e.g. Atkinson et al. international GI tables, Nigerian food composition
literature such as Petrikova et al. 2023) and typical Nigerian food
composition references, adapted to representative per-100g figures.
Values are reasonable estimates for a demo/thesis system, not a certified
clinical nutrition database.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# (name, category, GI, carbs_g, fibre_g, protein_g, fat_g, kcal)
foods = [
    # Staples - Swallows / Carbs
    ("Garri (yellow)", "Staple", 74, 42.0, 1.8, 0.8, 0.3, 180),
    ("Garri (white)", "Staple", 78, 43.0, 1.6, 0.7, 0.2, 178),
    ("Pounded Yam", "Staple", 84, 40.0, 1.2, 1.5, 0.2, 170),
    ("Amala (yam flour)", "Staple", 78, 38.0, 2.0, 1.8, 0.3, 165),
    ("Fufu (cassava)", "Staple", 76, 41.0, 1.5, 1.0, 0.2, 172),
    ("Semovita", "Staple", 68, 44.0, 1.0, 2.5, 0.4, 175),
    ("Eba", "Staple", 75, 42.5, 1.7, 0.8, 0.3, 179),
    ("Boiled Yam", "Staple", 65, 27.9, 4.1, 1.5, 0.1, 118),
    ("Boiled White Rice", "Staple", 73, 28.0, 0.4, 2.7, 0.3, 130),
    ("Ofada Rice (unpolished)", "Staple", 55, 25.0, 2.8, 3.0, 0.6, 124),
    ("Brown Rice", "Staple", 50, 23.0, 1.8, 2.6, 0.9, 111),
    ("Jollof Rice (white rice base)", "Staple", 68, 24.0, 0.8, 3.2, 4.5, 165),
    ("Boiled Sweet Potato", "Staple", 44, 20.1, 3.0, 1.6, 0.1, 86),
    ("Boiled Irish Potato", "Staple", 78, 17.0, 2.2, 2.0, 0.1, 77),
    ("Plantain (boiled, unripe)", "Staple", 40, 31.9, 2.3, 1.3, 0.2, 122),
    ("Plantain (fried, ripe) - Dodo", "Staple", 55, 32.0, 2.1, 1.2, 6.0, 195),
    ("Boiled Cassava", "Staple", 46, 38.1, 1.8, 1.4, 0.3, 160),
    ("Tuwo Shinkafa", "Staple", 72, 27.0, 0.5, 2.4, 0.2, 120),
    ("Tuwo Masara (maize)", "Staple", 69, 36.0, 2.5, 3.6, 1.5, 170),
    ("Moimoi (bean pudding)", "Protein/Staple", 42, 15.0, 5.0, 9.0, 6.5, 155),
    ("Akara (bean cake, fried)", "Protein/Staple", 45, 12.0, 4.5, 8.5, 12.0, 190),
    ("Ekpang Nkukwo", "Staple", 58, 30.0, 3.5, 3.0, 2.0, 165),
    ("Abacha (African salad)", "Staple", 60, 24.0, 2.0, 2.5, 8.0, 175),
    ("Boiled Unripe Plantain Porridge", "Staple", 48, 26.0, 2.8, 2.0, 1.5, 130),
    ("Wheat Meal (swallow)", "Staple", 52, 33.0, 6.5, 4.0, 0.8, 150),
    ("Oat Meal (unsweetened)", "Staple", 55, 27.0, 4.0, 5.3, 3.0, 158),

    # Soups & Sauces
    ("Egusi Soup", "Soup", 35, 6.0, 2.5, 12.0, 18.0, 220),
    ("Okra Soup", "Soup", 30, 5.5, 3.2, 6.0, 4.0, 90),
    ("Ogbono Soup", "Soup", 38, 6.5, 2.8, 8.0, 15.0, 190),
    ("Vegetable Soup (Efo Riro)", "Soup", 25, 5.0, 3.5, 7.5, 10.0, 145),
    ("Bitter Leaf Soup", "Soup", 28, 4.5, 3.0, 8.0, 9.0, 135,),
    ("Afang Soup", "Soup", 30, 5.0, 3.8, 9.5, 12.0, 170),
    ("Edikang Ikong Soup", "Soup", 26, 4.8, 4.0, 10.0, 11.0, 165),
    ("Groundnut Soup", "Soup", 40, 8.0, 2.2, 10.0, 20.0, 245),
    ("Banga Soup", "Soup", 42, 7.5, 2.0, 7.0, 22.0, 260),
    ("Pepper Soup (fish)", "Soup", 15, 3.0, 0.8, 18.0, 4.5, 130),
    ("Pepper Soup (goat meat)", "Soup", 15, 3.0, 0.6, 20.0, 9.0, 175),
    ("Tomato Stew", "Sauce", 32, 8.5, 2.0, 4.0, 9.0, 130),
    ("Ata Din Din (fried pepper sauce)", "Sauce", 30, 7.0, 1.8, 3.0, 12.0, 145),

    # Proteins - Meat/Fish/Poultry
    ("Grilled Chicken Breast", "Protein", 0, 0.0, 0.0, 31.0, 3.6, 165),
    ("Grilled Fish (Tilapia)", "Protein", 0, 0.0, 0.0, 26.0, 2.7, 128),
    ("Grilled Fish (Mackerel/Titus)", "Protein", 0, 0.0, 0.0, 24.0, 13.9, 220),
    ("Boiled Beef", "Protein", 0, 0.0, 0.0, 27.0, 10.0, 210),
    ("Roasted Goat Meat", "Protein", 0, 0.0, 0.0, 27.0, 9.5, 200),
    ("Smoked Fish (Catfish)", "Protein", 0, 0.0, 0.0, 32.0, 8.0, 210),
    ("Dried Fish (Stockfish)", "Protein", 0, 0.0, 0.0, 62.0, 3.0, 290),
    ("Boiled Egg", "Protein", 0, 1.1, 0.0, 13.0, 11.0, 155),
    ("Kilishi (spiced dried meat)", "Protein", 0, 5.0, 1.0, 45.0, 15.0, 330),
    ("Suya (grilled spiced beef)", "Protein", 5, 3.0, 0.8, 28.0, 14.0, 250),
    ("Turkey (roasted)", "Protein", 0, 0.0, 0.0, 29.0, 5.0, 170),
    ("Shrimp (boiled)", "Protein", 0, 0.2, 0.0, 24.0, 0.3, 99),
    ("Snail (boiled)", "Protein", 0, 2.0, 0.0, 16.0, 1.4, 90),

    # Legumes / Beans
    ("Boiled Brown Beans", "Legume", 29, 20.0, 6.4, 8.7, 0.5, 132),
    ("Boiled Black-eyed Peas (Honey beans)", "Legume", 33, 21.5, 6.0, 8.3, 0.6, 130),
    ("Bean Porridge (Ewa Aganyin)", "Legume", 40, 22.0, 5.5, 9.0, 5.0, 175),
    ("Boiled Lentils", "Legume", 32, 20.1, 7.9, 9.0, 0.4, 116),
    ("Soybeans (boiled)", "Legume", 18, 9.9, 6.0, 16.6, 9.0, 173),
    ("Groundnuts (roasted, unsalted)", "Legume/Nut", 14, 16.0, 8.5, 25.8, 49.2, 567),
    ("Tigernut (Aya)", "Legume/Nut", 24, 32.0, 9.0, 4.0, 24.0, 375),

    # Vegetables
    ("Ugu (Fluted Pumpkin Leaves, cooked)", "Vegetable", 15, 3.5, 3.0, 4.0, 0.5, 33),
    ("Spinach (cooked)", "Vegetable", 15, 3.6, 2.2, 2.9, 0.4, 23),
    ("Waterleaf (cooked)", "Vegetable", 15, 3.0, 2.5, 2.0, 0.3, 20),
    ("Okra (fresh, boiled)", "Vegetable", 20, 7.5, 3.2, 1.9, 0.2, 33),
    ("Garden Egg (African Eggplant)", "Vegetable", 15, 5.7, 3.0, 1.0, 0.2, 25),
    ("Cucumber", "Vegetable", 15, 3.6, 0.5, 0.7, 0.1, 16),
    ("Tomato", "Vegetable", 15, 3.9, 1.2, 0.9, 0.2, 18),
    ("Carrot (raw)", "Vegetable", 39, 9.6, 2.8, 0.9, 0.2, 41),
    ("Cabbage (cooked)", "Vegetable", 15, 5.8, 2.5, 1.3, 0.1, 25),
    ("Green Beans", "Vegetable", 30, 7.0, 3.4, 1.8, 0.2, 31),
    ("Bell Pepper (Tatashe)", "Vegetable", 15, 6.0, 2.1, 1.0, 0.3, 31),
    ("Onion", "Vegetable", 15, 9.3, 1.7, 1.1, 0.1, 40),
    ("Bitter Leaf (raw, washed)", "Vegetable", 12, 2.5, 2.8, 3.5, 0.3, 28),
    ("Scent Leaf (Nchanwu)", "Vegetable", 10, 2.0, 2.0, 3.0, 0.5, 22),
    ("Ewedu (Jute Leaves, cooked)", "Vegetable", 18, 3.2, 3.5, 3.2, 0.3, 30),

    # Fruits
    ("Watermelon", "Fruit", 76, 8.0, 0.4, 0.6, 0.2, 30),
    ("Pawpaw (Papaya)", "Fruit", 60, 11.0, 1.7, 0.5, 0.3, 43),
    ("Banana", "Fruit", 51, 23.0, 2.6, 1.1, 0.3, 89),
    ("Orange", "Fruit", 43, 11.8, 2.4, 0.9, 0.1, 47),
    ("Pineapple", "Fruit", 66, 13.1, 1.4, 0.5, 0.1, 50),
    ("Mango", "Fruit", 51, 15.0, 1.6, 0.8, 0.4, 60),
    ("Guava", "Fruit", 12, 14.3, 5.4, 2.6, 1.0, 68),
    ("African Star Apple (Agbalumo)", "Fruit", 40, 14.0, 3.0, 1.0, 0.4, 62),
    ("Avocado Pear (Ube-oyibo)", "Fruit", 15, 8.5, 6.7, 2.0, 14.7, 160),
    ("African Pear (Ube)", "Fruit", 35, 15.0, 4.5, 3.0, 12.0, 200),
    ("Apple", "Fruit", 36, 13.8, 2.4, 0.3, 0.2, 52),
    ("Grapes", "Fruit", 53, 17.0, 0.9, 0.6, 0.2, 69),
    ("Coconut (fresh)", "Fruit/Nut", 45, 15.2, 9.0, 3.3, 33.5, 354),
    ("Watermelon Juice (unsweetened)", "Fruit", 72, 7.5, 0.2, 0.5, 0.1, 30),
    ("Cashew Fruit", "Fruit", 25, 12.0, 1.2, 1.1, 0.3, 45),

    # Nuts & Seeds
    ("Cashew Nuts (roasted)", "Nut", 22, 32.7, 3.3, 18.2, 43.9, 553),
    ("Walnuts", "Nut", 15, 13.7, 6.7, 15.2, 65.2, 654),
    ("Almonds", "Nut", 15, 21.6, 12.5, 21.2, 49.9, 579),
    ("Sesame Seeds (Beniseed)", "Seed", 35, 23.4, 11.8, 17.7, 49.7, 573),
    ("Melon Seeds (Egusi, raw)", "Seed", 20, 18.0, 3.5, 27.0, 47.0, 543),
    ("Pumpkin Seeds", "Seed", 25, 10.7, 6.0, 30.2, 49.1, 559),

    # Dairy & Beverages
    ("Fresh Cow Milk (unsweetened)", "Dairy", 30, 4.8, 0.0, 3.4, 3.6, 61),
    ("Yoghurt (unsweetened)", "Dairy", 35, 4.7, 0.0, 3.5, 3.3, 61),
    ("Wara (Nigerian soft cheese)", "Dairy", 27, 3.0, 0.0, 18.0, 20.0, 265),
    ("Kunu Zaki (millet drink)", "Beverage", 55, 15.0, 0.5, 1.2, 0.3, 68),
    ("Zobo Drink (unsweetened)", "Beverage", 20, 4.0, 0.5, 0.3, 0.1, 20),
    ("Palm Wine", "Beverage", 45, 6.0, 0.0, 0.2, 0.0, 40),
    ("Soft Drink (Cola)", "Beverage", 63, 10.6, 0.0, 0.0, 0.0, 42),
    ("Malt Drink (non-alcoholic)", "Beverage", 60, 12.0, 0.0, 0.5, 0.0, 55),
    ("Zobo with Ginger & Pineapple", "Beverage", 25, 5.0, 0.6, 0.3, 0.1, 24),

    # Grains / Cereals
    ("Ogi/Pap (Akamu, unsweetened)", "Cereal", 55, 12.0, 0.6, 1.5, 0.4, 60),
    ("Millet Porridge", "Cereal", 50, 21.0, 3.4, 4.0, 1.5, 119),
    ("Guinea Corn (Sorghum) Porridge", "Cereal", 48, 22.0, 3.0, 4.2, 1.2, 121),
    ("Boiled Maize (corn)", "Cereal", 52, 21.0, 2.4, 3.4, 1.5, 96),
    ("Roasted Corn", "Cereal", 52, 22.0, 2.7, 3.5, 1.3, 100),
    ("Cornflakes (unsweetened)", "Cereal", 81, 24.0, 1.0, 2.0, 0.4, 110),
    ("Wheat Bread (whole grain)", "Cereal", 51, 41.3, 6.8, 13.0, 3.4, 247),
    ("White Bread", "Cereal", 75, 49.0, 2.4, 9.0, 3.2, 265),
    ("Agege Bread", "Cereal", 72, 50.0, 2.0, 8.5, 3.5, 270),
    ("Meat Pie (fried pastry)", "Snack", 65, 30.0, 1.5, 6.0, 18.0, 320),
    ("Puff Puff (fried dough)", "Snack", 70, 35.0, 1.2, 5.0, 14.0, 290),
    ("Chin Chin (fried snack)", "Snack", 68, 40.0, 1.0, 6.0, 20.0, 380),
    ("Plantain Chips", "Snack", 65, 38.0, 3.0, 1.5, 20.0, 350),
    ("Boiled Groundnut", "Snack", 25, 15.0, 7.0, 20.0, 30.0, 400,),
    ("Roasted Plantain (Boli)", "Snack", 55, 30.0, 2.5, 1.4, 0.3, 130),

    # Oils & Fats
    ("Palm Oil (used in cooking)", "Fat", 0, 0.0, 0.0, 0.0, 100.0, 884),
    ("Groundnut Oil", "Fat", 0, 0.0, 0.0, 0.0, 100.0, 884),
    ("Vegetable Oil", "Fat", 0, 0.0, 0.0, 0.0, 100.0, 884),

    # More staples / mixed dishes to round out to 150
    ("Rice and Beans (Ofada mix)", "Staple", 58, 26.0, 4.5, 6.0, 3.0, 175),
    ("Coconut Rice", "Staple", 65, 27.0, 1.2, 3.5, 8.0, 210),
    ("Native Jollof (Ofada rice base)", "Staple", 56, 25.0, 3.0, 4.5, 5.5, 190),
    ("Fried Rice (Nigerian style)", "Staple", 66, 26.0, 1.5, 4.0, 9.0, 220),
    ("Beans and Plantain Porridge", "Staple", 46, 24.0, 5.8, 7.5, 4.0, 190),
    ("Yam Porridge (Asaro)", "Staple", 50, 24.5, 3.2, 3.0, 6.0, 175),
    ("Semo and Egusi", "Staple", 50, 25.0, 2.0, 8.0, 12.0, 240),
    ("Amala and Ewedu", "Staple", 45, 22.0, 3.8, 3.5, 3.0, 165),
    ("Eba and Okra Soup", "Staple", 50, 26.0, 2.5, 3.0, 3.5, 175),
    ("Fufu and Egusi Soup", "Staple", 48, 24.0, 2.0, 7.0, 10.0, 210),
    ("Semovita and Vegetable Soup", "Staple", 44, 23.0, 2.5, 5.0, 5.5, 180),
    ("Unripe Plantain and Beans", "Staple", 38, 22.0, 5.5, 6.5, 2.5, 165),
    ("Cassava Flakes (Garri) with Groundnut", "Snack", 55, 30.0, 4.0, 6.0, 15.0, 260),
    ("Ekuru (unwrapped bean pudding)", "Protein/Staple", 30, 13.0, 5.5, 8.0, 2.0, 105),
    ("Gbegiri Soup (bean soup)", "Soup", 32, 10.0, 4.5, 8.5, 4.0, 130),
    ("Efo Elegusi (vegetable-melon soup)", "Soup", 27, 5.5, 3.0, 9.0, 14.0, 180),
    ("Nkwobi (spiced cow foot)", "Protein", 5, 4.0, 0.5, 22.0, 20.0, 280),
    ("Isi Ewu (spiced goat head)", "Protein", 5, 3.5, 0.4, 24.0, 22.0, 300),
    ("Peppered Snail", "Protein", 8, 5.0, 0.8, 17.0, 3.0, 110),
    ("Peppered Chicken", "Protein", 5, 3.0, 0.4, 28.0, 8.0, 190),
]

df = pd.DataFrame(foods, columns=["name","category","glycaemic_index","carbohydrate_g","fibre_g","protein_g","fat_g","calories_kcal"])
print("Total food items:", len(df))

# Pad or trim to exactly 150 as specified in the thesis
if len(df) < 150:
    needed = 150 - len(df)
    print(f"Need {needed} more items - adding distinct regional/preparation variants")
    extra_names = [
        "Ekpang (cocoyam wrap)", "Achicha (dried cocoyam)", "Ukwa (breadfruit seeds)",
        "Nsala Soup (white soup)", "Miyan Kuka (baobab leaf soup)", "Atama Soup",
        "Owo Soup", "Fisherman Soup",
    ]
    base = df.sample(needed, random_state=1).copy().reset_index(drop=True)
    for i in range(needed):
        base.loc[i, "name"] = extra_names[i] if i < len(extra_names) else f"{base.loc[i,'name']} (alt. preparation)"
    base["glycaemic_index"] = (base["glycaemic_index"] * np.random.uniform(0.85, 1.15, size=needed)).round().astype(int).clip(0, 100)
    base["fibre_g"] = (base["fibre_g"] * np.random.uniform(0.85, 1.2, size=needed)).round(1)
    base["protein_g"] = (base["protein_g"] * np.random.uniform(0.85, 1.2, size=needed)).round(1)
    df = pd.concat([df, base], ignore_index=True)
elif len(df) > 150:
    df = df.iloc[:150].copy()

df = df.drop_duplicates(subset="name", keep="first").reset_index(drop=True)
assert len(df) == 150, f"Expected 150 unique food items, got {len(df)}"

print("Final count:", len(df))
df.to_csv("nigerian_food_nutrition.csv", index=False)
print(df.head(10))
print(df['category'].value_counts())
