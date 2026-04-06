import pandas as pd


# First Create Basket:-
dataset = pd.read_csv("final_cleaned_data.csv")

Basket = (
    dataset.groupby(["Invoice","Description"])["Quantity"]
    .sum()
    .unstack()
    .fillna(0)
)

Basket = Basket.gt(0)

from mlxtend.frequent_patterns import apriori,association_rules
# step : -1 Frequent itemsets

frequent_items = apriori(Basket,min_support=0.03,use_colnames=True,max_len=3)

#  Set rules:-

rules = association_rules(frequent_items,metric="lift",min_threshold=1)

# Filters strong rules:-

best_rules = rules[
    (rules['lift'] > 1.5) &
    (rules['confidence'] > 0.60)
].sort_values(by='lift',ascending=False)

# print(best_rules)

# Convert frozenset → readable

best_rules['antecedents'] = best_rules['antecedents'].apply(lambda x: ','.join(list(x)))
best_rules['consequents'] = best_rules['consequents'].apply(lambda x: ','.join(list(x)))

# Print Top Combo:-

for i, row in best_rules.head(10).iterrows():
    print(f"If customer buy [{row['antecedents']}] --> Also buy [{row['consequents']}]")
    print(f"    confidence: {row['confidence']:.2f}, Lift : {row['lift']:.2f}")
    print("-"*60)


# Recommandation for user :-
def recommandation(product,rules_df):


    result = rules_df[rules_df['antecedents'].str.contains(product)]

    if result.empty:
        return "No recommendation found"
    

    return result[['antecedents', 'consequents', 'confidence', 'lift']]


print(recommandation("SWEETHEART CERAMIC TRINKET BOX", best_rules))


best_rules.to_csv("best_rules.csv",index = False)