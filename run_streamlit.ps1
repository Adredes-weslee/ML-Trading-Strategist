# PowerShell script to run the Streamlit application
Write-Host "Starting TradingStrategist application..." -ForegroundColor Green

# Get the Conda installation path - checking both Miniconda and Anaconda possible locations
$condaPaths = @(
    "$env:USERPROFILE\miniconda3",
    "$env:USERPROFILE\Anaconda3",
    "$env:USERPROFILE\.conda",
    "C:\ProgramData\Miniconda3",
    "C:\ProgramData\Anaconda3",
    "$env:LOCALAPPDATA\Continuum\miniconda3",
    "$env:LOCALAPPDATA\Continuum\anaconda3"
)

$condaPath = $null
foreach ($path in $condaPaths) {
    if (Test-Path $path) {
        $condaPath = $path
        Write-Host "Conda found at $condaPath" -ForegroundColor Green
        break
    }
}

if ($null -eq $condaPath) {
    Write-Host "Conda installation not found. Please install Conda or manually activate your environment before running this script." -ForegroundColor Red
    exit 1
}

# Find conda executable - it could be in different locations depending on the installation
$condaExe = $null
$possiblePaths = @(
    "$condaPath\Scripts\conda.exe",
    "$condaPath\condabin\conda.bat",
    "$condaPath\bin\conda.exe"
)

# Add more possible locations
$possiblePaths += @(
    "C:\ProgramData\Miniconda3\Scripts\conda.exe",
    "C:\ProgramData\Anaconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\Continuum\miniconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\Continuum\anaconda3\Scripts\conda.exe"
)

foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $condaExe = $path
        Write-Host "Found conda executable at $condaExe" -ForegroundColor Green
        break
    }
}

if ($null -eq $condaExe) {
    # If conda executable is not found, try a direct approach without conda
    Write-Host "Conda executable not found. Attempting direct approach..." -ForegroundColor Yellow
    
    # Environment name
    $envName = "trading-env"
    
    # Check if the environment exists
    $envPath = "$condaPath\envs\$envName"
    
    if (Test-Path $envPath) {
        Write-Host "Found environment at: $envPath" -ForegroundColor Green
        
        # Set PATH to use the Python from this environment directly
        $env:PATH = "$envPath;$envPath\Scripts;$envPath\Library\bin;$env:PATH"
        
        # Check if we can run python from this environment
        $pythonPath = "$envPath\python.exe"
        
        if (Test-Path $pythonPath) {
            Write-Host "Using Python from: $pythonPath" -ForegroundColor Green
            
            # Install required packages directly
            Write-Host "Installing required packages..." -ForegroundColor Yellow
            & $pythonPath -m pip install matplotlib streamlit pandas numpy pyyaml
            
            # Check if streamlit is installed in this environment
            $streamlitPath = & $pythonPath -c "import streamlit; print(streamlit.__file__)" 2>$null
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Found Streamlit at: $streamlitPath" -ForegroundColor Green
                
                # Run the app directly with this Python
                Write-Host "Running Streamlit application..." -ForegroundColor Green
                & $pythonPath -m streamlit run app.py
                exit 0
            } else {
                Write-Host "Streamlit not found in environment. Installing..." -ForegroundColor Yellow
                & $pythonPath -m pip install streamlit
                
                if ($LASTEXITCODE -eq 0) {
                    # Run the app after installation
                    Write-Host "Running Streamlit application..." -ForegroundColor Green
                    & $pythonPath -m streamlit run app.py
                    exit 0
                } else {
                    Write-Host "Failed to install Streamlit. Please install it manually with 'pip install streamlit'." -ForegroundColor Red
                    exit 1
                }
            }
        } else {
            Write-Host "Python not found in environment: $pythonPath" -ForegroundColor Red
        }
    } else {
        Write-Host "Environment not found: $envPath" -ForegroundColor Red
    }
    
    Write-Host "Failed to find or activate conda environment. Please ensure conda is properly installed." -ForegroundColor Red
    exit 1
}

# If we get here, we found conda executable and will use the standard approach
# Workaround for conda initialization in PowerShell
$envName = "trading-env"
Write-Host "Activating conda environment: $envName" -ForegroundColor Green

# Use the activate script directly
if (Test-Path "$condaPath\Scripts\activate.ps1") {
    & "$condaPath\Scripts\activate.ps1" $envName
} elseif (Test-Path "$condaPath\condabin\activate.ps1") {
    & "$condaPath\condabin\activate.ps1" $envName
} else {
    # If we can't find the activation script, try a direct approach
    Write-Host "Using direct approach to activate conda environment..." -ForegroundColor Yellow
    $env:PATH = "$condaPath\envs\$envName;$condaPath\envs\$envName\Library\mingw-w64\bin;$condaPath\envs\$envName\Library\usr\bin;$condaPath\envs\$envName\Library\bin;$condaPath\envs\$envName\Scripts;$condaPath\envs\$envName\bin;$env:PATH"
}

# Get python executable path
$pythonExe = "python"
if (Test-Path "$condaPath\envs\$envName\python.exe") {
    $pythonExe = "$condaPath\envs\$envName\python.exe"
    Write-Host "Using Python executable: $pythonExe" -ForegroundColor Green
}

# Install required packages directly using pip
Write-Host "Installing required packages..." -ForegroundColor Yellow
& $pythonExe -m pip install matplotlib streamlit pandas numpy pyyaml

# Check if streamlit is installed
$streamlitPath = & $pythonExe -c "import streamlit; print(streamlit.__file__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit is not installed in the current environment. Installing now..." -ForegroundColor Yellow
    & $pythonExe -m pip install streamlit
}
else {
    Write-Host "Using Streamlit from: $streamlitPath" -ForegroundColor Green
}

# Run the Streamlit application
Write-Host "Running Streamlit application..." -ForegroundColor Green
& $pythonExe -m streamlit run app.py