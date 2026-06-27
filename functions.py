# Suponha que 'potencia' é um array NumPy de 1 mês de leituras de um aparelho
def gera_dataset(path, disp, sens=15):
  df = pd.read_csv(path + "/" + disp)
  potencia = df["power"].to_numpy()
  classe_do_aparelho = disp.rsplit("_", 1)[0]
  tamanho_janela = 300
  meia_janela = tamanho_janela // 2
  
  # 1. Calcula a diferença de potência de um segundo para o outro
  diferencas = np.diff(potencia)

  # 2. Encontra os segundos onde houve transição (mudança > 15 Watts)
  indices_transicao = np.where(diferencas > sens)[0]

  # 3. Encontra os segundos onde o aparelho ficou estável/repouso (mudança < 2 Watts)
  indices_repouso = np.where(diferencas < 2)[0]

  # 4. Para extrair uma janela de transição:
  janelas_X = []
  janelas_Y = []

  idx_anterior = 0

  for idx in indices_transicao:
      # Garante que a janela não saia das bordas do array
      if idx > meia_janela and idx < (len(potencia) - meia_janela) and idx > (idx_anterior + tamanho_janela):
          # Janela centralizada no momento da mudança
          janela = potencia[idx - meia_janela : idx + meia_janela]
          janelas_X.append(janela)
          janelas_Y.append(classe_do_aparelho)
          idx_anterior = idx


  return janelas_X, janelas_Y


import torch
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def process_dataset(df):
  X = df.values[:,:-1].astype('float')
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  # substitui NaN/Inf se existir
  X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

  target = df.values[:,-1]

  encoder = LabelEncoder()
  y = encoder.fit_transform(target)

  n_input = X.shape[1]

  n_output = encoder.classes_.shape[0]


  X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.25,random_state=42)
  X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.2,random_state=42)

  train_dataset = CustomDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
  val_dataset = CustomDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
  test_dataset = CustomDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

  train_loader = DataLoader(dataset=train_dataset,batch_size=32, shuffle=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

  return train_loader, val_loader, test_loader



def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, patience=10, device="cpu", name=""):

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    counter = 0

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    for epoch in range(epochs):

        # ===== TREINO =====
        model.train()
        train_loss = 0

        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)


            optimizer.zero_grad()

            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # ===== accuracy =====
            _, preds = torch.max(outputs, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ===== VALIDAÇÃO =====
        model.eval()
        val_loss = 0

        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
              xb, yb = xb.to(device), yb.to(device)

              outputs = model(xb)
              loss = criterion(outputs, yb)
              val_loss += loss.item()
              # ===== accuracy =====
              _, preds = torch.max(outputs, 1)
              correct += (preds == yb).sum().item()
              total += yb.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # ===== EARLY STOPPING =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            # salva melhor modelo
            arq = "best_model_" + name + ".pth"
            torch.save(model.state_dict(), arq)

            es_epoch = epoch

        else:
            counter += 1

        # parar treino
        if counter >= patience:
            print(f"Early stopping na epoch {epoch}")
            break

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, es_epoch


import matplotlib.pyplot as plt

def show_train_grafic(train_losses, val_losses, es_epoch):
  
  plt.plot(train_losses, label="Train Loss")
  plt.plot(val_losses, label="Validation Loss")
  plt.axvline(x=es_epoch, linestyle='--', label='Early Stop')

  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title("Training vs Validation Loss")

  plt.legend()
  plt.grid()

  plt.show()


def predict(model, test_loader):
    model.eval()  # modo avaliação

    device = next(model.parameters()).device

    all_preds = []
    all_labels = []

    with torch.no_grad():  # não calcula gradiente
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)

            # pega classe com maior probabilidade
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    return all_preds, all_labels



def report(all_preds, all_labels):

  acc = accuracy_score(all_labels, all_preds)
  precision = precision_score(all_labels, all_preds, average="macro")
  recall = recall_score(all_labels, all_preds, average="macro")
  f1 = f1_score(all_labels, all_preds, average="macro")

  print(f"Acurácia: {acc:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall: {recall:.4f}")
  print(f"F1-score: {f1:.4f}")

  #print(classification_report(all_labels, all_preds))
  #cm = confusion_matrix(all_labels, all_preds)
  #print(cm)
  #acerto = 0
  #erro = 0
  #for i in range(len(y_pred)):
  #  if y_pred[i] == y_true[i]:
  #    acerto = acerto + 1
  #  else:
  #    erro = erro + 1
  #
  #print("Acertos: ", acerto)
  #print("Erros: ", erro)
  #print("Acurácia de %.2f %%" %(100 * acerto / (acerto + erro)))


def somar(a, b):
  return a + b
