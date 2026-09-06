"""
motor_combinatorio.py
======================
Motor matemático de inteligência combinatória para sistemas de loteria.
Implementação 100% vetorizada (NumPy) para:
  1. Geração da Matriz Principal (bilhetes) a partir de um universo reduzido de dezenas.
  2. Cálculo automático do Grupo Espelho (vetor complementar N \\ n).
  3. Validação por stress test Monte Carlo (10.000 simulações) sem laços for
     na etapa de conferência — usa produto matricial (np.dot) para checar
     todos os cartões contra todos os sorteios de uma vez.

NOTA TÉCNICA (importante para uso responsável):
A simulação mede a COBERTURA COMBINATÓRIA do conjunto de bilhetes, sob a premissa
de que o sorteio real cai dentro do universo escolhido pelo usuário (é assim que se
avalia a eficiência de um sistema de fechamento/wheeling). Os percentuais retornados
NÃO representam aumento da probabilidade real de acerto no sorteio oficial — cada
sorteio de loteria é um evento aleatório independente, e nenhuma estratégia de
escolha de números altera essa probabilidade matemática.
"""

import time
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


# ============================== Modelos (Request/Response) ==============================

class EngineRequest(BaseModel):
    universo: List[int] = Field(..., description="Dezenas escolhidas (universo reduzido), ex: 20 números")
    tamanho_cartao: int = Field(..., gt=0, description="Quantidade de dezenas por bilhete")
    universo_total: int = Field(60, gt=0, description="Tamanho total do universo da loteria (ex: 60 na Mega-Sena)")
    max_cartoes: int = Field(50, gt=0, description="Nº máximo de bilhetes gerados na Matriz Principal")
    num_simulacoes: int = Field(10_000, gt=0, description="Sorteios simulados no stress test")
    faixa_acerto_minima: int = Field(4, gt=0, description="Mínimo de acertos considerado 'sucesso'")

    @field_validator("universo")
    @classmethod
    def universo_nao_vazio(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("universo não pode ser vazio")
        if len(set(v)) != len(v):
            raise ValueError("universo não pode conter dezenas repetidas")
        return v


class DistribuicaoFaixa(BaseModel):
    acertos: int
    ocorrencias: int
    percentual: float


class EngineResponse(BaseModel):
    total_simulacoes: int
    total_sucessos: int
    taxa_eficiencia_percentual: float
    tempo_execucao_ms: float
    distribuicao_por_faixa: List[DistribuicaoFaixa]
    tamanho_matriz_principal: int
    tamanho_grupo_espelho: int
    grupo_espelho: List[int]


# ============================== Núcleo Matemático ==============================

def gerar_grupo_espelho(universo: List[int], universo_total: int) -> List[int]:
    """Calcula o vetor complementar (Grupo Espelho): dezenas excluídas do universo escolhido."""
    todas_dezenas = set(range(1, universo_total + 1))
    return sorted(todas_dezenas - set(universo))


def gerar_matriz_principal(universo: List[int], tamanho_cartao: int, max_cartoes: int) -> List[Tuple[int, ...]]:
    """Gera os bilhetes (combinações) do universo escolhido, limitado a max_cartoes."""
    if tamanho_cartao > len(universo):
        raise ValueError("tamanho_cartao não pode ser maior que o universo escolhido")
    universo_ordenado = sorted(universo)
    cartoes = []
    for i, combo in enumerate(combinations(universo_ordenado, tamanho_cartao)):
        if i >= max_cartoes:
            break
        cartoes.append(combo)
    if not cartoes:
        raise ValueError("Não foi possível gerar bilhetes com os parâmetros informados")
    return cartoes


def _binarizar_cartoes(cartoes: List[Tuple[int, ...]], universo_total: int) -> np.ndarray:
    """Converte bilhetes em matriz binária (n_cartoes x universo_total) para conferência vetorizada."""
    matriz = np.zeros((len(cartoes), universo_total), dtype=np.int8)
    for idx, cartao in enumerate(cartoes):
        indices = np.array(cartao) - 1  # dezenas 1..N -> índices 0..N-1
        matriz[idx, indices] = 1
    return matriz


def simular_sorteios(universo: List[int], tamanho_cartao: int, universo_total: int,
                      num_simulacoes: int) -> np.ndarray:
    """
    Gera N sorteios aleatórios (sem reposição) dentro do universo escolhido, de forma
    totalmente vetorizada (sem for por simulação): usa argsort de chaves aleatórias
    para simular embaralhamento em massa.
    """
    universo_arr = np.array(sorted(universo)) - 1
    tamanho_universo = len(universo_arr)

    chaves_aleatorias = np.random.rand(num_simulacoes, tamanho_universo)
    ordem = np.argsort(chaves_aleatorias, axis=1)
    escolhidos_local = ordem[:, :tamanho_cartao]
    escolhidos_global = universo_arr[escolhidos_local]

    matriz_sorteios = np.zeros((num_simulacoes, universo_total), dtype=np.int8)
    linhas = np.repeat(np.arange(num_simulacoes), tamanho_cartao)
    matriz_sorteios[linhas, escolhidos_global.flatten()] = 1
    return matriz_sorteios


def conferir_acertos_vetorizado(matriz_cartoes: np.ndarray, matriz_sorteios: np.ndarray) -> np.ndarray:
    """
    Confere TODOS os cartões contra TODOS os sorteios em uma única operação matricial
    (np.dot) — zero laços for. Retorna o melhor nº de acertos (entre os cartões da
    Matriz Principal) para cada sorteio simulado.
    """
    acertos_matrix = np.dot(matriz_cartoes, matriz_sorteios.T)  # (n_cartoes, num_simulacoes)
    return acertos_matrix.max(axis=0)


def calcular_distribuicao(melhores_acertos: np.ndarray, tamanho_cartao: int) -> List[Dict[str, Any]]:
    """Monta a distribuição por faixa de acertos (0 até tamanho_cartao), vetorizado."""
    total = len(melhores_acertos)
    faixas = np.arange(tamanho_cartao + 1)
    ocorrencias = (melhores_acertos[:, None] == faixas[None, :]).sum(axis=0)
    percentuais = np.round((ocorrencias / total) * 100, 4) if total else np.zeros_like(ocorrencias, dtype=float)
    return [
        {"acertos": int(f), "ocorrencias": int(o), "percentual": float(p)}
        for f, o, p in zip(faixas, ocorrencias, percentuais)
    ]


# ============================== Orquestração / Pipeline ==============================

def rodar_engine(payload: EngineRequest) -> Dict[str, Any]:
    inicio = time.perf_counter()

    grupo_espelho = gerar_grupo_espelho(payload.universo, payload.universo_total)
    cartoes = gerar_matriz_principal(payload.universo, payload.tamanho_cartao, payload.max_cartoes)
    matriz_cartoes = _binarizar_cartoes(cartoes, payload.universo_total)
    matriz_sorteios = simular_sorteios(
        payload.universo, payload.tamanho_cartao, payload.universo_total, payload.num_simulacoes
    )

    melhores_acertos = conferir_acertos_vetorizado(matriz_cartoes, matriz_sorteios)
    distribuicao = calcular_distribuicao(melhores_acertos, payload.tamanho_cartao)

    total_sucessos = int(np.sum(melhores_acertos >= payload.faixa_acerto_minima))
    taxa_eficiencia = round((total_sucessos / payload.num_simulacoes) * 100, 4)

    tempo_execucao_ms = round((time.perf_counter() - inicio) * 1000, 3)

    return {
        "total_simulacoes": payload.num_simulacoes,
        "total_sucessos": total_sucessos,
        "taxa_eficiencia_percentual": taxa_eficiencia,
        "tempo_execucao_ms": tempo_execucao_ms,
        "distribuicao_por_faixa": distribuicao,
        "tamanho_matriz_principal": len(cartoes),
        "tamanho_grupo_espelho": len(grupo_espelho),
        "grupo_espelho": grupo_espelho,
    }


# ============================== API FastAPI ==============================

app = FastAPI(
    title="Motor de Inteligência Combinatória de Loterias",
    version="1.0.0",
    description="Engine vetorizado para geração de bilhetes, grupo espelho e stress test Monte Carlo.",
)


@app.post("/engine/simular", response_model=EngineResponse)
def simular(payload: EngineRequest):
    try:
        return rodar_engine(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


# ============================== Execução direta (benchmark local) ==============================

if __name__ == "__main__":
    exemplo = EngineRequest(
        universo=list(range(1, 21)),   # 20 dezenas escolhidas
        tamanho_cartao=6,
        universo_total=60,
        max_cartoes=50,
        num_simulacoes=10_000,
        faixa_acerto_minima=4,
    )
    resultado = rodar_engine(exemplo)
    print("Tempo de execução (ms):", resultado["tempo_execucao_ms"])
    print("Taxa de eficiência (%):", resultado["taxa_eficiencia_percentual"])
    print("Total de sucessos:", resultado["total_sucessos"], "/", resultado["total_simulacoes"])
    print("Tamanho grupo espelho:", resultado["tamanho_grupo_espelho"], "->", resultado["grupo_espelho"])
    print("Distribuição:", resultado["distribuicao_por_faixa"])
