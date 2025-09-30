# Radiotherapy QA Assistant

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-ready--for--testing-yellow.svg)

AI-assisted quality assurance tools for radiotherapy treatment plans and medical physics measurements. This project provides Streamlit-based web interfaces for automated QA checks following standard protocols (TG-142, TG-224) and monitor unit calculations.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows (PowerShell) or Linux/Mac
- CSV files with measurement data

### Installation

```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/radiotherapy-qa-assistant.git
cd radiotherapy-qa-assistant

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Validate installation
python validate_system.py
```

### Usage

**📊 QA Analysis Interface:**
```powershell
streamlit run app.py
```
- Upload CSV with measurement data
- Select QA modules (TG142, TG224, Electron Beam, TPS)
- Review pass/fail results
- Download Excel reports

**⚡ Monitor Unit Calculator:**
```powershell
streamlit run dose_calculator_streamlit.py
```
- Calculate MU for given parameters
- Compare with TPS values
- Validate within tolerance limits

**📋 Batch Processing:**
```powershell
python "python phase1_qa_tool.py"
```

## 📁 Project Structure

```
├── app.py                      # Main Streamlit QA interface
├── dose_calculator_streamlit.py # MU calculator interface
├── qa_checks.py               # Core QA function implementations
├── app_utils.py               # Utility functions and data processing
├── sample_data.csv            # Example measurement data
├── requirements.txt           # Python dependencies
├── validate_system.py         # Installation validation script
├── QUICKSTART.md             # Quick reference guide
├── TESTING_GUIDE.md          # Comprehensive testing instructions
└── tests/                    # Test suite

```

## 🔬 QA Modules

### TG-142 Checks
- **Output**: ±2% tolerance
- **Flatness**: ±3% tolerance  
- **Symmetry**: ±3% tolerance

### TG-224 Checks
- **Dose Uniformity**: ±2% tolerance
- **Beam Energy**: ±1 MV tolerance

### Electron Beam Checks
- **Electron Output**: ±3% tolerance
- **PDD**: ±2% tolerance

### TPS Comparison
- **Planned vs Measured**: ±3% tolerance
- **Gamma Index**: ≥95% pass rate

## 📊 Data Format

Your CSV files should include relevant columns:

```csv
output,flatness,symmetry,dose_uniformity,beam_energy,electron_output,PDD,planned_vs_measured,gamma_index
100.0,99.5,100.2,100.1,6.0,101.0,99.8,98.5,97.2
98.5,102.1,99.8,99.5,6.1,99.2,100.5,101.2,96.8
```

## 🧪 Testing

**Quick validation:**
```powershell
python validate_system.py
```

**Run test suite:**
```powershell
python -m pytest tests/ -v
```

**Manual testing:**
See `TESTING_GUIDE.md` for comprehensive testing procedures for medical physicists.

## 🛠️ For Developers

### Adding New QA Modules

1. Create function in `qa_checks.py`
2. Return list of dicts with keys: `Sample`/`index`, `parameter`, `status`, `value`
3. Add to UI selection in `app.py`
4. Update tests in `tests/`

### Integration with External Analysis

Use `integration_IC23.ipynb` as a template for incorporating custom analysis code (e.g., IC23 width calculations).

## 📚 Documentation

- **`QUICKSTART.md`** - Essential commands and daily usage
- **`TESTING_GUIDE.md`** - Comprehensive testing procedures
- **`.github/copilot-instructions.md`** - AI agent guidance for development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is intended for research and educational purposes. Always validate results against established procedures and consult with qualified medical physicists before clinical use.

## 🏥 Medical Physics Community

Built for medical physicists, by medical physicists. Feedback and contributions from the medical physics community are welcome!

---

**Status: Ready for clinical testing and validation**
