import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from datetime import datetime

# ML / Analytics
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛍",
    layout="wide"
)

# ---------------- FUNCTIONS ----------------

def prepare_dataset(dataset):
    dataset.columns = dataset.columns.str.strip()

    if "Total_amount" not in dataset.columns:
        if "Quantity" in dataset.columns and "Price" in dataset.columns:
            dataset["Total_amount"] = dataset["Quantity"] * dataset["Price"]

    dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])

    return dataset


def generate_rfm(dataset):
    snapshot_date = dataset['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = dataset.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'Total_amount': 'sum'
    })

    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'Total_amount': 'Monetary'
    }, inplace=True)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm)

    cluster_names = {
        0: "Regular Customers",
        1: "Premium Customers",
        2: "At Risk",
        3: "New Customers"
    }

    rfm['cluster_name'] = rfm['cluster'].map(cluster_names)

    rfm.reset_index(inplace=True)

    return rfm


def generate_rules(dataset):
    # Create Basket exactly like your standalone script
    basket = (
        dataset
        .groupby(['Invoice', 'Description'])['Quantity']
        .sum()
        .unstack()
        .fillna(0)
    )

    # Convert to boolean matrix (modern + fast)
    basket = basket.gt(0)

    # Step 1: Frequent Itemsets (same parameters as your code)
    frequent_items = apriori(
        basket,
        min_support=0.03,
        use_colnames=True,
        max_len=3
    )

    if frequent_items.empty:
        return pd.DataFrame()

    # Step 2: Generate Rules
    rules = association_rules(
        frequent_items,
        metric="lift",
        min_threshold=1
    )

    if rules.empty:
        return pd.DataFrame()

    # Step 3: Filter Strong Rules (same as your logic)
    best_rules = rules[
        (rules['lift'] > 1.5) &
        (rules['confidence'] > 0.60)
    ].sort_values(by='lift', ascending=False)

    if best_rules.empty:
        return pd.DataFrame()

    # Convert frozenset to readable text (important for dashboard display)
    best_rules['antecedents'] = best_rules['antecedents'].apply(
        lambda x: ', '.join(list(x))
    )

    best_rules['consequents'] = best_rules['consequents'].apply(
        lambda x: ', '.join(list(x))
    )

    return best_rules


# ---------------- DATA SOURCE ----------------
st.sidebar.header("📂 Data Source")


data_option = st.sidebar.radio(
    "Choose Dataset",
    ["Use Default Dataset", "Upload New Dataset"]
)

if data_option == "Upload New Dataset":

    with st.sidebar.expander("📘 View Sample Dataset Format"):

        st.markdown("""
        Your uploaded CSV should contain the following columns:

        - Invoice  
        - Description  
        - Quantity  
        - Price  
        - Customer ID  
        - Country  
        - InvoiceDate
        """)

        sample_data = pd.DataFrame({
            "Invoice": [10001, 10001, 10002],
            "Description": ["Milk", "Bread", "Butter"],
            "Quantity": [2, 1, 3],
            "Price": [50, 30, 60],
            "Customer ID": [12345, 12345, 67890],
            "Country": ["United Kingdom", "United Kingdom", "France"],
            "InvoiceDate": ["2024-01-01", "2024-01-01", "2024-01-02"]
        })

        st.dataframe(sample_data)

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        dataset = pd.read_csv(uploaded_file)
        dataset = prepare_dataset(dataset)

        # Train models on uploaded dataset
        rfm = generate_rfm(dataset)
        best_rules = generate_rules(dataset)

        st.session_state.dataset = dataset
        st.session_state.rfm = rfm
        st.session_state.best_rules = best_rules

        st.sidebar.success("Dataset uploaded and models trained successfully")

    else:
        st.stop()

else:

    dataset = pd.read_pickle(
        "dataset_small.pkl",
        compression="gzip"
    )

    dataset = prepare_dataset(dataset)

    rfm = generate_rfm(dataset)
    best_rules = generate_rules(dataset)


# Use session data if exists
if "dataset" in st.session_state:
    dataset = st.session_state.dataset

if "rfm" in st.session_state:
    rfm = st.session_state.rfm

if "best_rules" in st.session_state:
    best_rules = st.session_state.best_rules


# ---------------- NAVIGATION BAR ----------------

selected = option_menu(
    menu_title=None,
    options=[
        "Home",
        "EDA Analysis",
        "RFM + Recommendation",
        "Best Combos",
    ],
    icons=[
        "house",
        "bar-chart",
        "person",
        "fire"
    ],
    orientation="horizontal"
)


# ---------------- HOME PAGE ----------------

if selected == "Home":

    st.title("🛍 Retail Analytics Dashboard")

    st.markdown(
        "Welcome to the **Retail Intelligence System**"
    )

    st.divider()

    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Customers",
        dataset['Customer ID'].nunique()
    )

    col2.metric(
        "🧾 Orders",
        dataset['Invoice'].nunique()
    )

    col3.metric(
        "📦 Products",
        dataset['Description'].nunique()
    )

    col4.metric(
        "💰 Revenue",
        f"$ {int(dataset['Total_amount'].sum())}"
    )

    st.divider()

    st.subheader("📄 Dataset Preview")

    st.dataframe(dataset.head())


# ---------------- EDA PAGE ----------------

elif selected == "EDA Analysis":

    st.title("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Top Selling Products (Revenue)")

        top_product = (
            dataset.groupby('Description')["Total_amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(top_product)

    with col2:
        st.write("Most Purchased Products (Quantity)")

        top_quantity = (
            dataset.groupby('Description')["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(top_quantity)

    st.divider()

    sales_trend = (
        dataset.set_index('InvoiceDate')
        .resample('M')['Total_amount']
        .sum()
    )

    st.subheader("📈 Monthly Sales Trend")

    st.line_chart(sales_trend)


# ---------------- RFM PAGE ----------------

elif selected == "RFM + Recommendation":

    st.title(
        "👤 Customer Segmentation + Recommendation"
    )

    if not rfm.empty:

        customer_id = st.selectbox(
            "Select Customer",
            rfm['Customer ID']
        )

        customer = rfm[
            rfm["Customer ID"] == customer_id
        ]

        segment = customer[
            "cluster_name"
        ].values[0]

        st.subheader(
            f"Segment: {segment}"
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(
                "📊 Customer Details"
            )

            st.write(customer)

        with col2:
            st.subheader(
                "🛒 Smart Recommendation"
            )

            customer_history = dataset[
                dataset["Customer ID"]
                == customer_id
            ]

            past_products = (
                customer_history[
                    "Description"
                ]
                .value_counts()
                .head(3)
            )

            for item in past_products.index:
                st.markdown(
                    f"👉 **{item}**"
                )

    else:

        st.warning(
            "No customers found"
        )


# ---------------- BEST COMBOS PAGE ----------------

elif selected == "Best Combos":

    st.title("🔥 Best Product Combinations")

    if not best_rules.empty:

        top_rules = (
            best_rules
            .sort_values(
                by='lift',
                ascending=False
            )
            .head(10)
        )

        for i, row in top_rules.iterrows():

            st.success(
                f"👉 {row['antecedents']} → {row['consequents']}"
            )

            st.write(
                f"Confidence: {row['confidence']:.2f} | Lift: {row['lift']:.2f}"
            )

            st.divider()

    else:

        st.warning(
            "No association rules generated"
        )
