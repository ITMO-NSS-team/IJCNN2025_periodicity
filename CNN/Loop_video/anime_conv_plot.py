import pandas as pd
from matplotlib import pyplot as plt

combs = [(2, 8), (4, 8), (8, 8)]
for comb in combs:
    prehistory = comb[0]
    forecast = comb[1]
    folder = f'models_anime_{prehistory}_{forecast}'
    df = pd.read_csv(f'{folder}/anime_90000.csv')
    plt.plot(df['epoch'], df['loss'], label=f'input frames={prehistory}, '
                                            f'forecast size={forecast}')
plt.legend()
plt.yscale('log')
plt.title('Looped data train loss')
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.tight_layout()
plt.savefig('anime_convergence.png', dpi=400)
plt.show()