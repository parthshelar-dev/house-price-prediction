import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

if "screen" not in st.session_state:
    st.session_state["screen"] = "home"

if st.session_state["screen"] == "home":

    st.title("House Price Predictor 🏠")
    st.caption("Built using Linear Regression, Ridge Regression, and Random Forest — this app automatically selects the best model to give you the most accurate house price prediction.")
    st.divider()

    st.subheader("Select your option")
    st.write("Choose whether you're buying or selling to see price predictions.")

    col1, col2 = st.columns(2)

    with col1:
        st.info("🏠 **Buying a House?**\n\nEnter property details and find out the fair market value before making an offer.")
        if st.button("Start Buying →", use_container_width=True):
            st.session_state["screen"] = "buy"
            st.rerun()

    with col2:
        st.info("💰 **Selling a House?**\n\nEnter your property details and get the recommended listing price.")
        if st.button("Start Selling →", use_container_width=True):
            st.session_state["screen"] = "sell"
            st.rerun()
    
    st.stop()

if st.session_state["screen"] == "buy":
    
    if st.button("← Back"):
        st.session_state["screen"] = "home"
        st.rerun()

    st.title("🏠 Buy a House")
    st.caption("Enter the property details below to find out the fair market value.")
    st.divider()

    st.subheader("🏡 Property Details")

    col1, col2 = st.columns(2)

    with col1:
        
        overall_qual = st.slider("Overall Quality", 1, 10, 5)
        st.caption("1 = Very poor, 10 = Excellent finish")

        GarageCars = st.slider("Garage Capacity", 0, 4, 0)
        st.caption("Number of cars the garage can fit.")

        FullBath = st.slider("Number of Bathrooms", 0, 4, 1)
        st.caption("Number of bathrooms in the house.")

    with col2:
        
        GrLivArea = st.number_input("Living Area (sq ft)", min_value=300, value=1500)
        st.caption("Total above-ground living space.")

        TotalBsmtSF = st.number_input("Basement Area (sq ft)", min_value=0, value=800)
        st.caption("Enter 0 if there is no need of basement.")

        YearBuilt = st.number_input("Year Built", min_value=1872, max_value=2024, value=2000)
        st.caption("Originally constructed year.")

    st.divider()

    if st.button("🔍 Predict Fair Value", use_container_width=True):
        with st.spinner("Analysing property details..."):
            BASE_DIR = Path(__file__).parent
            model = joblib.load(BASE_DIR / "model.pkl")
            scaler = joblib.load(BASE_DIR / "scaler.pkl")
            poly = joblib.load(BASE_DIR / "poly.pkl")
            best_model_name = joblib.load(BASE_DIR / "best_model_name.pkl")

            current_year = datetime.now().year
            HouseAge = current_year - YearBuilt

            user_data = pd.DataFrame([[
                overall_qual, GrLivArea, GarageCars,
                TotalBsmtSF, FullBath, HouseAge
            ]], columns=[
                "OverallQual", "GrLivArea", "GarageCars",
                "TotalBsmtSF", "FullBath", "HouseAge"
            ])

            if best_model_name == "RandomForest":
                price = model.predict(user_data)[0]
            else:
                user_poly = poly.transform(user_data)
                user_scaled = scaler.transform(user_poly)
                price = model.predict(user_scaled)[0]

        st.success("Analysis Complete!")
        st.metric(label="Fair Market Value", value=f"${price:,.2f}")
        st.caption(f"Model used: {best_model_name}")
        st.info("💡 This is the fair market value. Consider making an offer at or below this price.")

        st.divider()
        st.subheader("📊 Property Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Quality", f"{overall_qual}/10")
        c2.metric("Living Area", f"{GrLivArea} sqft")
        c3.metric("House Age", f"{HouseAge} yrs")

        c4, c5, c6 = st.columns(3)
        c4.metric("Bathrooms", f"{FullBath}")
        c5.metric("Garage", f"{GarageCars} cars")
        c6.metric("Basement", f"{TotalBsmtSF} sqft")
    
    st.stop()

if st.session_state["screen"] == "sell":

    if st.button("← Back"):
        st.session_state["screen"] = "home"
        st.rerun()

    st.title("💰 Sell a House")
    st.caption("Enter your property details below to get the recommended listing price.")
    st.divider()

    st.subheader("🏡 Property Details")

    col1, col2 = st.columns(2)

    with col1:
        
        overall_qual = st.slider("Overall Quality", 1, 10, 5)
        st.caption("1 = Very poor, 10 = Excellent finish")

        GarageCars = st.slider("Garage Capacity", 0, 4, 0)
        st.caption("Number of cars the garage can fit.")

        FullBath = st.slider("Number of Bathrooms", 0, 4, 1)
        st.caption("NUmber of bathrooms in the house.")

    with col2:
        
        GrLivArea = st.number_input("Living Area (sq ft)", min_value=300, value=1500)
        st.caption("Total above-ground living space.")

        TotalBsmtSF = st.number_input("Basement Area (sq ft)", min_value=0, value=800)
        st.caption("Enter 0 if there is no need of basement.")

        YearBuilt = st.number_input("Year Built", min_value=1872, max_value=2024, value=2000)
        st.caption("Originally constructed year.")

    st.divider()

    if st.button("🔍 Get Listing Price", use_container_width=True):
        with st.spinner("Calculating best listing price..."):
            BASE_DIR = Path(__file__).parent
            model = joblib.load(BASE_DIR / "model.pkl")
            scaler = joblib.load(BASE_DIR / "scaler.pkl")
            poly = joblib.load(BASE_DIR / "poly.pkl")
            best_model_name = joblib.load(BASE_DIR / "best_model_name.pkl")

            current_year = datetime.now().year
            HouseAge = current_year - YearBuilt

            user_data = pd.DataFrame([[
                overall_qual, GrLivArea, GarageCars,
                TotalBsmtSF, FullBath, HouseAge
            ]], columns=[
                "OverallQual", "GrLivArea", "GarageCars",
                "TotalBsmtSF", "FullBath", "HouseAge"
            ])

            if best_model_name == "RandomForest":
                price = model.predict(user_data)[0]
            else:
                user_poly = poly.transform(user_data)
                user_scaled = scaler.transform(user_poly)
                price = model.predict(user_scaled)[0]

            low_price = price * 0.95
            high_price = price * 1.05

        st.success("Listing Price Ready!")
        st.metric(label="Recommended Listing Price", value=f"${price:,.2f}")
        st.caption(f"Model used: {best_model_name}")

        col1, col2 = st.columns(2)
        col1.metric("List Low At", f"${low_price:,.2f}", delta="-5%")
        col2.metric("List High At", f"${high_price:,.2f}", delta="+5%")

        st.info("💡 List within this range for the best chance of a quick and profitable sale.")

        st.divider()
        st.subheader("📊 Property Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Quality", f"{overall_qual}/10")
        c2.metric("Living Area", f"{GrLivArea} sqft")
        c3.metric("House Age", f"{HouseAge} yrs")

        c4, c5, c6 = st.columns(3)
        c4.metric("Bathrooms", f"{FullBath}")
        c5.metric("Garage", f"{GarageCars} cars")
        c6.metric("Basement", f"{TotalBsmtSF} sqft")

    st.stop()