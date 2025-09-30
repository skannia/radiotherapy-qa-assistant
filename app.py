import io
import streamlit as st
import pandas as pd
from qa_checks import tg142_check, tg224_check, electron_beam_check, tps_check
from app_utils import normalize_uploaded_df

st.title("Phase 1 QA Tool")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Normalize uploaded DataFrame (unique columns, ensure Sample)
    df, info = normalize_uploaded_df(df)
    if info.get('renamed'):
        st.warning('Uploaded CSV contained duplicate columns; duplicates were renamed to make headers unique.')
    st.write("Sample Data:", df)

    # Let user choose QA modules
    selected_qas = st.multiselect(
        "Select QA types to run",
        options=["TG142", "TG224", "Electron Beam", "TPS"],
        default=["TG142"]
    )

    # Run selected QA modules
    results = []
    if "TG142" in selected_qas:
        results += tg142_check(df)
    if "TG224" in selected_qas:
        results += tg224_check(df)
    if "Electron Beam" in selected_qas:
        results += electron_beam_check(df)
    if "TPS" in selected_qas:
        results += tps_check(df)

    # Pivot results for readability
    if results:
        df_report = pd.DataFrame(results).pivot(index='Sample', columns='parameter', values='status')
        df_report.reset_index(inplace=True)
        st.write("QA Report:", df_report)

        # Safe Excel download: write to a BytesIO buffer and pass bytes to Streamlit
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_report.to_excel(writer, index=False, sheet_name='Report')
            try:
                if 'info' in locals() and info.get('renamed'):
                    import pandas as _pd
                    renames = info.get('renames')
                    meta_df = _pd.DataFrame([{'original': v, 'new': k} for k, v in renames.items()])
                    meta_df.to_excel(writer, index=False, sheet_name='Metadata')
            except Exception:
                pass
        buffer.seek(0)
        st.download_button(
            "Download Excel Report",
            data=buffer.getvalue(),
            file_name="qa_report_readable.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.info("Select at least one QA type to run.")
