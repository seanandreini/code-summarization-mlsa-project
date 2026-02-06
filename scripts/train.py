import argparse

def main():
  parser = argparse.ArgumentParser(description="Code Summarization Training Script")

  parser.add_argument(
    '--model',
    type=str,
    required=True,
    choices=['transformer', 'lstm']
  )

  # common args
  parser.add_argument('--train_samples', type=int)
  parser.add_argument('--epochs', type=int,)
  parser.add_argument('--batch_size', type=int)
  parser.add_argument('--lr', type=float)
  parser.add_argument('--n_layers', type=int)
  parser.add_argument('--patience', type=int)
  parser.add_argument('--dropout', type=float)
  
  # transformer args
  parser.add_argument('--n_heads', type=int)
  parser.add_argument('--d_model', type=int)
  parser.add_argument('--ff_units', type=int)

  # lstm args
  parser.add_argument('--embedding_dim', type=int)
  parser.add_argument('--hidden_dim', type=int)
  parser.add_argument('--teacher_forcing_prob', type=float)



  # assert d_model % h == 0, f"Errore: d_model ({d_model}) deve essere divisibile per n_heads ({h})!"