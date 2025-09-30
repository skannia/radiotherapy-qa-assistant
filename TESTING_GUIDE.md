# Testing Instructions for Medical Physicists
**AI-Assisted QA for Radiotherapy Treatment Plans**

## Quick Start Guide

### Prerequisites
- Windows computer with Python 3.8+ installed
- Basic familiarity with running commands in PowerShell
- CSV files with measurement data for testing

### 1. Setup (One-time installation)

Open PowerShell and navigate to the project folder:

```powershell
# Navigate to the project directory
cd "c:\path\to\your\project\folder"

# Create a virtual environment (recommended)
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

**Note:** If you get an execution policy error, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Testing the Main QA Application

**Start the QA Tool:**
```powershell
streamlit run app.py
```

This opens a web browser with the QA interface. Test with these steps:

1. **Upload test data**: Use the provided `sample_data.csv` or your own CSV file
2. **Select QA modules**: Choose from TG142, TG224, Electron Beam, or TPS checks
3. **Run analysis**: Click to execute the selected QA checks
4. **Review results**: Check the pass/fail status for each parameter
5. **Download report**: Save the Excel report for documentation

**Expected columns in your CSV:**
- `output`, `flatness`, `symmetry` (for TG142)
- `dose_uniformity`, `beam_energy` (for TG224)
- `electron_output`, `PDD` (for Electron Beam)
- `planned_vs_measured`, `gamma_index` (for TPS)

### 3. Testing the MU Calculator

**Start the MU Calculator:**
```powershell
streamlit run dose_calculator_streamlit.py
```

Test scenarios:
1. **Basic calculation**: Enter prescribed dose (200 cGy), dose rate (1.0 cGy/MU), depth (10 cm)
2. **TPS comparison**: Enter TPS MU value and check deviation percentage
3. **Tolerance check**: Verify the ±3% tolerance validation works correctly

### 4. Testing the Batch Processing Script

**Run the example script:**
```powershell
python "python phase1_qa_tool.py"
```

This should:
- Generate `sample_data.csv` with test data
- Run all QA modules
- Create `qa_report_readable.xlsx` with pivoted results

### 5. Testing the Integration Notebook (Optional)

**Start Jupyter notebook:**
```powershell
# Install jupyter if not already installed
pip install jupyter

# Start notebook server
jupyter notebook integration_IC23.ipynb
```

**In the notebook:**
1. Run Cell 1 to import libraries and test app_utils integration
2. Run Cell 2 to test the placeholder IC23 width calculation
3. Replace the placeholder function with your actual IC23 analysis code

## Validation Checklist

### ✅ Core Functionality Tests

**QA Module Tests:**
- [ ] TG142 checks: output (±2%), flatness (±3%), symmetry (±3%)
- [ ] TG224 checks: dose uniformity (±2%), beam energy (±1 MV)
- [ ] Electron beam: output (±3%), PDD (±2%)
- [ ] TPS comparison: planned vs measured (±3%), gamma index (≥95%)

**MU Calculator Tests:**
- [ ] Basic MU calculation produces reasonable values
- [ ] TPS comparison shows percentage deviation correctly
- [ ] Tolerance checking (±3%) works for pass/fail determination

**Data Processing Tests:**
- [ ] CSV upload handles your data format correctly
- [ ] Missing columns are handled gracefully
- [ ] Results pivot correctly into readable Excel format

### ⚠️ Common Issues and Solutions

**"Module not found" errors:**
- Ensure virtual environment is activated
- Verify all packages installed: `pip list`

**"File not found" errors:**
- Check file paths are correct
- Ensure CSV files are in the expected location

**Streamlit won't start:**
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

**Unicode/encoding errors:**
- Ensure CSV files are saved in UTF-8 format
- Check that measurement data contains only numeric values

### 📊 Test Data Requirements

**Minimum test dataset:**
- At least 3 measurement records
- All required columns present with numeric values
- Values within reasonable ranges for medical physics measurements

**Example test values:**
```
output: 98-102 (percent)
flatness: 97-103 (percent)  
symmetry: 97-103 (percent)
dose_uniformity: 98-102 (percent)
beam_energy: 5.5-6.5 (MV)
electron_output: 97-103 (percent)
PDD: 98-102 (percent)
planned_vs_measured: 97-103 (percent)
gamma_index: 90-99 (percent)
```

### 📝 Testing Report Template

**For each test session, document:**
1. Date and tester name
2. Software version/commit hash
3. Test data used (file names, number of records)
4. Results summary (pass/fail counts)
5. Any issues encountered
6. Performance observations (speed, memory usage)
7. Suggestions for improvement

### 🔄 Workflow Integration Testing

Test the complete workflow:
1. **Data preparation**: Import measurement data from your systems
2. **QA execution**: Run appropriate checks for your protocols
3. **Results review**: Validate against known good measurements
4. **Documentation**: Generate reports for clinical records
5. **Trending**: Test with historical data to identify patterns

### 📞 Support and Feedback

When reporting issues, include:
- Complete error messages
- Steps to reproduce the problem
- Sample data files (de-identified)
- Your system configuration (Windows version, Python version)

This tool is designed to integrate with existing medical physics QA workflows. Test thoroughly with your specific measurement protocols and data formats before clinical use.