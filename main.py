import subprocess
import sys

def main():
    print("INICIANDO O PIPELINE MLOPS VIA DVC")
    
    try:
        # Executa o comando 'dvc repro' no terminal
        result = subprocess.run(["dvc", "repro"], check=True)
        
        print("Para ver as métricas, execute: dvc metrics show")
        
    except subprocess.CalledProcessError:
        print("\nOcorreu um erro durante a execução do pipeline.")
        sys.exit(1)

if __name__ == "__main__":
    main()