import pandas as pd

if __name__ == '__main__':
    result_latex = pd.read_csv('results/results.csv').rename(columns={'Unnamed: 0': 'Attack-Dataset-Model'}).to_latex(index=False)
    begin = '\\begin{table}[h!]\n\\centering'
    end = '\\caption{Caption}\n\\label{tab:my_label}\n\\end{table}'
    with open('results/latex_table.tex', 'w') as f:
        f.write(begin)
        f.write(result_latex)
        f.write(end)
