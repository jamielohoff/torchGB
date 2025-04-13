import torch


def angle_between_tensors(tensor1, tensor2):
  """
  Computes the angle between two pytorch tensors.

  Args:
    tensor1: The first pytorch tensor.
    tensor2: The second pytorch tensor.

  Returns:
    The angle between the two tensors in radians.
  """
  return torch.atan2(torch.sum(tensor1 * tensor2), torch.sum(tensor1) * torch.sum(tensor2))


def cosine_similarity(tensor1, tensor2):
  """
  Computes the cosine similarity between two PyTorch tensors.

  Args:
    tensor1: The first pytorch tensor.
    tensor2: The second pytorch tensor.

  Returns:
    The cosine similarity between the two tensors.
  """
  return torch.sum(tensor1 * tensor2) / (torch.norm(tensor1) * torch.norm(tensor2))

