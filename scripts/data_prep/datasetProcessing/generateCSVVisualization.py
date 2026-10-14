import os
import pandas as pd
import glob

def processar_arquivos_emocao(pasta_origem, arquivo_saida='emocoes.csv'):
    """
    Processa arquivos .txt com probabilidades emocionais (valores separados por vírgula)
    e gera um CSV
    
    Args:
        pasta_origem (str): Caminho da pasta com os arquivos .txt
        arquivo_saida (str): Nome do arquivo CSV de saída
    """
    
    # Lista de estados emocionais (na ordem em que aparecem nos arquivos)
    estados_emocionais = [
        'happy', 'contempt', 'elated', 'hopeful', 'surprised', 
        'proud', 'loved', 'angry', 'astonished', 'disgusted', 
        'fearful', 'sad', 'fatigued', 'neutral'
    ]
    
    # Lista para armazenar os dados
    dados = []
    
    # Encontrar todos os arquivos .txt na pasta
    padrao_arquivos = os.path.join(pasta_origem, '*_prob_rank.txt')
    arquivos_txt = glob.glob(padrao_arquivos)
    
    print(f"Encontrados {len(arquivos_txt)} arquivos .txt")
    
    for caminho_arquivo in arquivos_txt:
        try:
            # Extrair nome do arquivo do caminho completo
            nome_arquivo = os.path.basename(caminho_arquivo)
            
            # Ler o conteúdo do arquivo
            with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
                conteudo = arquivo.read().strip()
            
            # Dividir os valores por vírgula
            valores_str = conteudo.split(',')
            
            # Converter para float e limpar espaços em branco
            valores = []
            for valor_str in valores_str:
                valor_limpo = valor_str.strip()
                if valor_limpo:  # Se não for string vazia
                    try:
                        valor = float(valor_limpo)
                        valores.append(valor)
                    except ValueError:
                        print(f"Aviso: Valor não numérico encontrado em {nome_arquivo}: '{valor_limpo}'")
            
            # Verificar se temos exatamente 14 valores
            if len(valores) != 14:
                print(f"Aviso: Arquivo {nome_arquivo} tem {len(valores)} valores (esperados: 14)")
                print(f"Valores encontrados: {valores}")
                continue
            
            # Criar dicionário com os dados
            registro = {}
            
            # Adicionar as probabilidades para cada estado emocional
            for i, estado in enumerate(estados_emocionais):
                registro[estado] = valores[i]
            
            # Adicionar o caminho do arquivo
            registro['file_path'] = caminho_arquivo
            
            dados.append(registro)
            
        except Exception as e:
            print(f"Erro ao processar arquivo {caminho_arquivo}: {e}")
    
    # Criar DataFrame
    df = pd.DataFrame(dados)
    
    # Reordenar colunas para colocar file_path por último
    colunas = estados_emocionais + ['file_path']
    df = df[colunas]
    
    # Salvar como CSV
    df.to_csv(arquivo_saida, index=False)
    print(f"Arquivo CSV salvo como: {arquivo_saida}")
    print(f"Total de registros processados: {len(dados)}")
    
    return df

# Versão alternativa sem pandas para valores separados por vírgula
def processar_sem_pandas(pasta_origem, arquivo_saida='emocoes.csv'):
    """
    Versão alternativa sem usar pandas para valores separados por vírgula
    """
    
    estados_emocionais = [
        'happy', 'contempt', 'elated', 'hopeful', 'surprised', 
        'proud', 'loved', 'angry', 'astonished', 'disgusted', 
        'fearful', 'sad', 'fatigued', 'neutral'
    ]
    
    # Encontrar arquivos
    padrao_arquivos = os.path.join(pasta_origem, '*_prob_rank.txt')
    arquivos_txt = glob.glob(padrao_arquivos)
    
    with open(arquivo_saida, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Escrever cabeçalho
        cabecalho = estados_emocionais + ['file_path']
        writer.writerow(cabecalho)
        
        contador = 0
        for caminho_arquivo in arquivos_txt:
            try:
                with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
                    conteudo = arquivo.read().strip()
                
                # Dividir valores por vírgula
                valores_str = conteudo.split(',')
                
                # Converter para float
                valores = []
                for valor_str in valores_str:
                    valor_limpo = valor_str.strip()
                    if valor_limpo:
                        try:
                            valores.append(float(valor_limpo))
                        except ValueError:
                            # Se não conseguir converter, pular este arquivo
                            print(f"Valor inválido em {caminho_arquivo}: '{valor_limpo}'")
                            valores = []
                            break
                
                if len(valores) == 14:
                    linha_csv = valores + [caminho_arquivo]
                    writer.writerow(linha_csv)
                    contador += 1
                else:
                    print(f"Arquivo {caminho_arquivo} tem {len(valores)} valores (esperados: 14)")
                
            except Exception as e:
                print(f"Erro em {caminho_arquivo}: {e}")
    
    print(f"CSV gerado: {arquivo_saida}")
    print(f"Registros processados: {contador}")

# Exemplo de uso
if __name__ == "__main__":
    # Substitua pelo caminho da sua pasta
    pasta_com_arquivos = "C:\\Users\\joaoc\\AffectNetFusedDifferent\\val_set\\annotations"  # Altere para o caminho correto
    
    # Verificar se a pasta existe
    if not os.path.exists(pasta_com_arquivos):
        print(f"Pasta '{pasta_com_arquivos}' não encontrada!")
        print("Por favor, altere a variável 'pasta_com_arquivos' para o caminho correto.")
    else:
        # Processar arquivos e gerar CSV
        df_resultado = processar_arquivos_emocao(pasta_com_arquivos)
        
        # Mostrar as primeiras linhas do resultado
        print("\nPrimeiras 5 linhas do DataFrame:")
        print(df_resultado.head())
        
        # Mostrar informações básicas do dataset
        print(f"\nInformações do dataset:")
        print(f"Shape: {df_resultado.shape}")
        print(f"Colunas: {list(df_resultado.columns)}")