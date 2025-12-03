# Virtual Environment Setup Guide

## Quick Start

### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Windows (Command Prompt)
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Linux/Mac
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Detailed Instructions

### 1. Create Virtual Environment

**What it does:** Creates an isolated Python environment for your project.

```powershell
python -m venv venv
```

You can name it anything (e.g., `env`, `myenv`), but `venv` is standard.

### 2. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

**You'll see `(venv)` at the beginning of your prompt when activated.**

### 3. Upgrade pip (Recommended)

```powershell
python -m pip install --upgrade pip
```

### 4. Install Project Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Verify Installation

```powershell
pip list
```

---

## Common Issues & Solutions

### PowerShell Execution Policy Error

**Error:** `cannot be loaded because running scripts is disabled`

**Solution:**
```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy for current user (recommended)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate again
.\venv\Scripts\Activate.ps1
```

### Python Not Found

**Solution:**
- Ensure Python is installed: `python --version`
- Try `python3` instead of `python`
- Add Python to PATH

### Wrong Python Version

**Solution:** Specify the Python version explicitly
```powershell
# Use specific Python version
py -3.12 -m venv venv
```

---

## Working with Virtual Environment

### Installing Additional Packages

```powershell
# Install a single package
pip install package-name

# Install and add to requirements
pip install package-name
pip freeze > requirements.txt
```

### Deactivating Virtual Environment

```powershell
deactivate
```

### Removing Virtual Environment

Simply delete the `venv` folder:
```powershell
# Make sure it's deactivated first
deactivate

# Remove the folder
Remove-Item -Recurse -Force venv
```

---

## Alternative: Using Conda

### Create Conda Environment

```bash
# Create environment with specific Python version
conda create -n myenv python=3.12

# Activate
conda activate myenv

# Install from requirements.txt
pip install -r requirements.txt

# Or install with conda
conda install numpy pandas scikit-learn matplotlib seaborn jupyter

# Deactivate
conda deactivate
```

---

## VS Code Integration

### 1. Select Python Interpreter

1. Press `Ctrl+Shift+P` (Command Palette)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `.\venv\Scripts\python.exe`

### 2. Automatic Activation

VS Code will automatically activate your venv when you:
- Open a terminal
- Run Python files
- Use Jupyter notebooks

### 3. Settings (Optional)

Add to `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}\\venv\\Scripts\\python.exe",
    "python.terminal.activateEnvironment": true
}
```

---

## Best Practices

1. **Always activate** before working on the project
2. **Never commit** the `venv` folder (it's in `.gitignore`)
3. **Update requirements.txt** when adding packages:
   ```powershell
   pip freeze > requirements.txt
   ```
4. **Document** any special installation steps in README.md
5. **Use same Python version** across team members

---

## Quick Reference

| Task | Command (Windows PowerShell) |
|------|------------------------------|
| Create venv | `python -m venv venv` |
| Activate | `.\venv\Scripts\Activate.ps1` |
| Deactivate | `deactivate` |
| Install packages | `pip install -r requirements.txt` |
| Install package | `pip install package-name` |
| Update requirements | `pip freeze > requirements.txt` |
| List packages | `pip list` |
| Upgrade pip | `python -m pip install --upgrade pip` |

---

## Troubleshooting

### Check if venv is active
```powershell
# Should show path to venv
Get-Command python | Select-Object -ExpandProperty Source
```

### Reinstall all packages
```powershell
pip install -r requirements.txt --force-reinstall
```

### Create requirements from scratch
```powershell
pip freeze > requirements.txt
```

### Use specific pip version
```powershell
python -m pip install package-name
```

---

## Next Steps

After setting up your virtual environment:

1. ✓ Activate virtual environment
2. ✓ Install dependencies
3. ✓ Configure VS Code interpreter
4. Start working on your project!
5. Run notebooks: `jupyter notebook`
6. Run scripts: `python scripts/run_pipeline.py`
