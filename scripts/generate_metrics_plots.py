import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_plots(csv_path=r"/media/prometheus/Arquivos/Faculdade/UCP/5 periodo/projeto_final/pcc/results/result11m9.csv", output_dir="."):
    """
    Script para Geração dos Gráficos Oficiais do PCC
    Lê o histórico de treinamento (result11m9.csv) e gera 3 figuras em alta resolução (300 DPI).
    """
    print(f"Lendo histórico de treinamento de: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Ajustar estilos visuais
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8

    epochs = df['epoch']

    # -------------------------------------------------------------
    # Gráfico 1: Curvas de Aprendizado e Métricas Globais
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.plot(epochs, df['metrics/mAP50(B)'], label='mAP@0.5 (Pico: 79,58%)', color='#0072b2', linewidth=2.5)
    ax.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95 (Pico: 52,07%)', color='#009e73', linewidth=2.0)
    ax.plot(epochs, df['metrics/recall(B)'], label='Recall / Sensibilidade (Pico: 85,05%)', color='#d55e00', linewidth=1.8, linestyle='--')
    ax.plot(epochs, df['metrics/precision(B)'], label='Precisão Global (Pico: 73,86%)', color='#cc79a7', linewidth=1.8, linestyle=':')

    # Ponto de pico no mAP50
    peak_epoch = df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch']
    peak_map = df['metrics/mAP50(B)'].max()
    ax.scatter([peak_epoch], [peak_map], color='#0072b2', s=70, zorder=5)
    ax.annotate(f'Pico mAP@0.5: {peak_map*100:.2f}%\n(Época {int(peak_epoch)})',
                xy=(peak_epoch, peak_map), xytext=(peak_epoch-25, peak_map-0.12),
                arrowprops=dict(facecolor='#0072b2', shrink=0.08, width=1, headwidth=6),
                fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f4f8', edgecolor='#0072b2'))

    ax.set_title('Convergência das Métricas Globais de Desempenho do SafetyVision (YOLO11m)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Épocas de Treinamento', fontsize=10)
    ax.set_ylabel('Pontuação da Métrica (0.0 a 1.0)', fontsize=10)
    ax.set_ylim(0.1, 0.9)
    ax.set_xlim(1, max(epochs))
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plot1_path = f"{output_dir}/grafico_curvas_aprendizado_metricas.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Salvo: {plot1_path}")

    # -------------------------------------------------------------
    # Gráfico 2: Perdas de Treinamento e Validação
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    # Box Loss
    axes[0].plot(epochs, df['train/box_loss'], label='Treino', color='#0072b2', linewidth=2)
    axes[0].plot(epochs, df['val/box_loss'], label='Validação', color='#d55e00', linewidth=2, linestyle='--')
    axes[0].set_title('Perda de Localização (Box Loss)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Perda (Loss)')
    axes[0].legend()

    # Class Loss
    axes[1].plot(epochs, df['train/cls_loss'], label='Treino', color='#0072b2', linewidth=2)
    axes[1].plot(epochs, df['val/cls_loss'], label='Validação', color='#d55e00', linewidth=2, linestyle='--')
    axes[1].set_title('Perda de Classificação (Class Loss)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Épocas')
    axes[1].legend()

    # DFL Loss
    axes[2].plot(epochs, df['train/dfl_loss'], label='Treino', color='#0072b2', linewidth=2)
    axes[2].plot(epochs, df['val/dfl_loss'], label='Validação', color='#d55e00', linewidth=2, linestyle='--')
    axes[2].set_title('Perda Focal de Distribuição (DFL Loss)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Épocas')
    axes[2].legend()

    plt.suptitle('Evolução Comparativa das Funções de Perda - Treino vs. Validação (Estabilidade sem Overfitting)', fontsize=12, fontweight='bold', y=1.03)
    plt.tight_layout()
    plot2_path = f"{output_dir}/grafico_perdas_treino_validacao.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Salvo: {plot2_path}")

    # -------------------------------------------------------------
    # Gráfico 3: Trade-off F1-Score e Taxa de Aprendizado (LR)
    # -------------------------------------------------------------
    p = df['metrics/precision(B)']
    r = df['metrics/recall(B)']
    f1 = 2 * (p * r) / (p + r + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Subplot 1: F1-Score
    ax1.plot(epochs, f1, color='#009e73', linewidth=2.2)
    max_f1_idx = f1.idxmax()
    ax1.scatter([epochs[max_f1_idx]], [f1[max_f1_idx]], color='#009e73', s=60, zorder=5)
    ax1.annotate(f'Pico F1: {f1[max_f1_idx]*100:.2f}%\n(Época {epochs[max_f1_idx]})',
                 xy=(epochs[max_f1_idx], f1[max_f1_idx]),
                 xytext=(epochs[max_f1_idx]-30, f1[max_f1_idx]-0.15),
                 arrowprops=dict(facecolor='#009e73', shrink=0.08, width=1, headwidth=5),
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#009e73'))
    ax1.set_title('Evolução do F1-Score Médio (Equilíbrio P e R)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Épocas de Treinamento')
    ax1.set_ylabel('F1-Score (0.0 a 1.0)')
    ax1.set_ylim(0.3, 0.85)

    # Subplot 2: Learning Rate
    ax2.plot(epochs, df['lr/pg0'], color='#e69f00', linewidth=2)
    ax2.set_title('Programação da Taxa de Aprendizado (Cosine Annealing)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Épocas de Treinamento')
    ax2.set_ylabel('Taxa de Aprendizado (LR)')

    plt.suptitle('Desempenho Harmônico F1-Score e Agendamento do Otimizador', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot3_path = f"{output_dir}/grafico_tradeoff_f1_lr.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Salvo: {plot3_path}")

if __name__ == "__main__":
    generate_plots()
