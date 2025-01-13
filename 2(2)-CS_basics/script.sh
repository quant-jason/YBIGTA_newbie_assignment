#!/bin/bash

# 1. Miniconda 설치 여부 확인 및 설치
if ! command -v conda &> /dev/null; then
    echo "Miniconda is not installed. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    echo "Miniconda installed."
else
    echo "Miniconda is already installed."
fi

# 2. 가상환경 활성화
echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda create -y -n my_env python=3.8
conda activate my_env

# 3. submission 폴더의 Python 파일 실행
echo "Executing Python files..."
for file in submission/*.py; do
    base_name=$(basename "$file" .py)  # 파일 이름에서 확장자 제거
    input_file="input/${base_name}_input"  # input 디렉토리의 입력 파일 경로
    output_file="output/${base_name}_output"  # output 디렉토리의 출력 파일 경로
    
    if [[ -f "$input_file" ]]; then
        echo "Running $file with input $input_file..."
        python "$file" < "$input_file" > "$output_file"
        echo "Output saved to $output_file."
    else
        echo "Input file $input_file does not exist. Skipping $file."
    fi
done

# 4. mypy 테스트 수행
echo "Running mypy tests..."
for file in submission/*.py; do
    echo "Testing $file..."
    mypy "$file"
done

echo "Script execution complete."