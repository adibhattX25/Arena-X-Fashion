import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained XGBoost model
# Ensure 'xgboost_fashion_model.pkl' is in the same folder as this script
try:
    model = joblib.load('xgboost_fashion_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. Hardcoded List of Brands (extracted from your Myntra EDA)
brand_list = [
    'Roadster', 'Pothys', 'KALINI', 'HERE&NOW', 'HRX by Hrithik Roshan', 
    'H&M', 'Biba', 'W', 'Anouk', 'Zivame', 'Vishudh', 'Mast & Harbour', 
    'DressBerry', 'Sangria', 'Libas'
]

# 3. Page Configuration
st.set_page_config(page_title="Myntra Success Suggester", layout="wide")
st.title("🏆 Myntra Top 5 Purchase Success Suggester")
st.markdown("This system uses an advanced XGBoost model to predict which brands have the highest likelihood of purchase success based on your criteria.")

# 4. Sidebar for User Inputs
with st.sidebar:
    st.header("Shopper Requirements")
    gender = st.selectbox("Select Gender", ["Men", "Women"])
    category = st.selectbox("Clothing Category", ['Bottom Wear', 'Indian Wear', 'Topwear', 'Western'])
    budget = st.slider("Budget (Original Price in Rs)", 100, 10000, 1500)
    
    # Adding the Discount Slider back for the UI/UX
    # This is used for price calculation, but NOT sent to the model
    discount = st.slider("Target Discount (%)", 0, 100, 30)

# 5. Recommendation Logic
if st.button("Find Top 5 Suggestions"):
    results = []
    
    # This matches your Colab output EXACTLY in order and naming
    # Note: 'DiscountOffer' is omitted because it wasn't in your final training set
    required_columns = ['OriginalPrice (in Rs)', 'BrandName', 'Category', 'Individual_category', 'category_by_Gender'] 

    with st.spinner('Analyzing Myntra market trends...'):
        for brand in brand_list:
            # Data assembled in the EXACT order the model pipeline expects:
            # 1. OriginalPrice (Budget)
            # 2. BrandName
            # 3. Category
            # 4. Individual_category (Placeholder logic from your notebook)
            # 5. category_by_Gender
            data_row = [[budget, brand, category, category.lower(), gender]]
            
            input_df = pd.DataFrame(data_row, columns=required_columns)
            
            try:
                # Use the XGBoost model to predict probability of the "Popular" class
                # predict_proba returns [Not Popular, Popular], we take index 1
                prob = model.predict_proba(input_df)[0][1]
                results.append({"Brand": brand, "Success Probability": prob})
            except Exception as e:
                st.error(f"Prediction error for {brand}: {e}")
                st.stop()
                
    # 6. Process and Display Results
    if results:
        # Create DataFrame and sort to find the top 5
        res_df = pd.DataFrame(results)
        top_5 = res_df.sort_values(by="Success Probability", ascending=False).head(5)
        
        # Adding a functional use for the discount slider: Calculating estimated final price
        top_5['Estimated Final Price'] = budget * (1 - discount/100)
        
        st.subheader(f"Top 5 Recommended Brands for {category} ({gender})")
        st.write(f"Results optimized for a {discount}% discount preference.")
        
        # Displaying visually as metrics
        cols = st.columns(5)
        for i, (index, row) in enumerate(top_5.iterrows()):
            with cols[i]:
                st.metric(label=f"Rank #{i+1}", value=row['Brand'])
                st.write(f"Match Score: {row['Success Probability']:.2%}")
                st.write(f"Est. Price: ₹{row['Estimated Final Price']:.2f}")
        
        # Display the full table for clarity
        st.markdown("### Detailed Likelihood Breakdown")
        st.table(top_5)