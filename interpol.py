import pandas as pd

# Load csv file into a DataFrame
df = pd.read_csv('interp_input.csv', parse_dates=[0], index_col=0, dayfirst=True)
df.drop(columns=['Mokro', 'Loznica'], inplace=True)

# --------- Substitute stations where possible ---------
transfer = {
            'Mateševo': 'Kolašin',
            'Velika': 'Plav',
            'Štitarica': 'Mojkovac',
            'Pošcenje': 'Šavnik'
           }
df.rename(columns=transfer, inplace=True)

# --------- Interpolate to get stations that have no substitutes -----------
# Define the distances
distances = {
    ('Andrijevica', 'Berane'): 14311.5,
    ('Andrijevica', 'Plav'): 15277.5,
    ('Andrijevica', 'Kolašin'): 19970.5,
    ('Rožaje', 'Berane'): 23001,
    ('Rožaje', 'Plav'): 25927,
    ('Goražde', 'Foča'): 22683.5,
    ('Goražde', 'Višegrad'): 29343,
}

# Function to calculate the weighted value
def calc_inv_dist_weight(row, target, distances):
    weights = []
    values = []
    for (location1, location2), distance in distances.items():
        if location1 == target and location2 in df.columns:
            weight = 1 / (distance ** 2)
            weights.append(weight)
            values.append(row[location2] * weight)
    if weights:
        return sum(values) / sum(weights)
    return None

# Create new columns for the target locations
targets = ['Andrijevica', 'Goražde', 'Rožaje']
for target in targets:
    df[target] = df.apply(lambda row: calc_inv_dist_weight(row, target, distances), axis=1)

df.to_csv('interp_output.csv', encoding='utf-8-sig', index=False)

