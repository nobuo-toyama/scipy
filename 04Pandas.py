# =============================================
#    Pandas
# =============================================
# =============================================
#    Getting Started
# =============================================

# ==== What kind of data does pandas handle?
import pandas as pd


df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr, Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
    }
)
df
df["Age"]
ages = pd.Series([22, 35, 58], name="Age")
ages

df["Age"].max()
ages.max()

df.describe()

# ==== How do I read and write tabular data?
morning = pd.read_csv("Morning.csv")
morning
morning.describe()
morning.head(8)
morning.dtypes
morning.info()

# ==== How do I select a subset of a DataFrame?
pulse = morning["Pulse"]
pulse.head()
type(morning["Pulse"])
morning["Pulse"].shape
pulse_high = morning[["Pulse", "High"]]
pulse_high.head()
morning[["Pulse", "High"]].shape

after_2020 = morning[morning["Date"] > "2020-12-31"]
after_2020.head()
after_2020.shape

stage_1 = morning[(morning["High"] > 130) | (morning["Low"] > 80)]
stage_1.head(10)

high_no_na = morning[morning["High"].notna()]
high_no_na.tail()

high_low = morning.loc[morning["High"] > 150, "Low"]
high_low.head()

morning.iloc[60:70, :3]

# ==== How to create plots in pandas?
import pandas as pd
import matplotlib.pyplot as plt

morning.plot()
morning["Pulse"].plot()
morning.plot.scatter(x="High", y="Low", alpha=0.5)

morning.plot.box()
axs = morning.plot.area(figsize=(12, 4), subplots=True)

fig, axs = plt.subplots(figsize=(12, 4))
morning.plot.area(ax=axs)
axs.set_ylabel("mmHg")
fig.savefig("blood_pressure.png")

# ==== How to create new columns derived from existing columns?
morning["diff"] = morning["High"] - morning["Low"]
morning.head()
morning_renamed = morning.rename(
    columns={"diff": "Diff"}
)
morning_renamed.head()

# ==== How to calculate summary statistics?
morning["High"].mean()
morning[["High", "Low"]].median()
morning[["High", "Low"]].describe()
morning["Pulse"].value_counts()

# ==== How to reshape the layout of tables?
# Sort table rows
morning.sort_values(by="High").head()
morning.sort_values(by=["High", "Low"], ascending=False).head()
# Long to wide table format

# ==== How to combine data from multiple tables?
# Join tables using a common identifier
evening = pd.read_csv("Evening.csv")
evening.head()

blood_pressure = pd.merge(morning, evening, how="left", on="Date")
blood_pressure.head()

# ==== How to handle time series data with ease?
morning = pd.read_csv("Morning.csv", parse_dates=["Date"])
morning.head()
morning["Date"].min(), morning["Date"].max()
morning["Date"].max() - morning["Date"].min()

morning["month"] = morning["Date"].dt.month
morning.head()
morning.tail()

morning.groupby(morning["Date"].dt.weekday)["High"].mean()

fig, axs = plt.subplots(figsize=(12, 4))
morning.groupby(morning["Date"].dt.weekday)["High"].mean().plot(kind="bar", rot=0, ax=axs)
plt.xlabel("Day of the week")
plt.ylabel("mmHg")
