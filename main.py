import streamlit as st
import torch
from utilities import process_text_regex

from utilities import load_model, get_prediction


def main():
    model = load_model()
    print('model loaded successfully')
    st.title("Text Processing Application")

    input_text = st.text_area("Enter your text:")
    input_text = input_text.strip().upper()

    method_selected = None
    cols = st.columns(2)

    if cols[0].button("Process with Regex Method"):
        method_selected = "Regex Method"
    if cols[1].button("Process with LSTM Method"):
        method_selected = "LSTM Method"

    if method_selected:
        try:
            if method_selected == "Regex Method":
                output = process_text_regex(input_text)
                st.success(f"Regex Method: Number of consecutive CGs are :  {output}")
            elif method_selected == "LSTM Method":
                print('Herrrrrrrrrrrrrrrrrrrrrrrrrreeeeeeeeeeeeeeeeee')
                output = get_prediction(model, input_text)
                st.success(f"LSTM Method: Number of consecutive CGs are :  {output}")
        except Exception as e:
            st.error(f"An error occurred with {method_selected}: {e}")

if __name__ == "__main__":
    main()

