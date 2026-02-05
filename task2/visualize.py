import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def plot_confusion(y_true, preds):
    cm = confusion_matrix(y_true,preds)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot()
    plt.title("Confusion Matrix")
    plt.show()


import matplotlib.pyplot as plt

def plot_probs_vs_true(probs, y_true, threshold=0.5):
    pairs = sorted(zip(probs, y_true), key=lambda x: x[0])
    
    probs_sorted, y_sorted = zip(*pairs)
    

    plt.figure(figsize=(10, 6))
    
    plt.plot(probs_sorted, label="Proqnoz Ehtimalı (Probability)", color='blue')
 
    plt.scatter(range(len(y_sorted)), y_sorted, color='orange', alpha=0.5, label="Real Etiket (True Label)", s=10)
    

    plt.axhline(y=threshold, color='red', linestyle='--', label=f"Hədd ({threshold})")
    
 
    plt.title("Proqnoz Ehtimalı vs Real Nəticə (Sıralanmış)")
    plt.xlabel("Nümunələr (Ehtimala görə azdan çoxa)")
    plt.ylabel("Ehtimal / Sinif")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()