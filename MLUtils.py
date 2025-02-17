{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMxkPos6L2uC+w6/nzZCVZP",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sngo/learnpytorch/blob/main/MLUtils.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DrTDQ_AJlhY2"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HUzk4-aVuhHM",
        "outputId": "15257895-0172-4d03-bff9-076b777ed8bd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "PyTorch version: 2.5.1+cu124\n",
            "torchvision version: 0.20.1+cu124\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(None, 'cpu')"
            ]
          },
          "metadata": {},
          "execution_count": 1
        }
      ],
      "source": [
        "# Import PyTorch\n",
        "import torch\n",
        "from torch import nn\n",
        "\n",
        "# Import torchvision\n",
        "import torchvision\n",
        "from torchvision import datasets\n",
        "from torchvision.transforms import ToTensor\n",
        "\n",
        "# Import matplotlib for visualization\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Check versions\n",
        "# Note: your PyTorch version shouldn't be lower than 1.10.0 and torchvision version shouldn't be lower than 0.11\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "print(f\"PyTorch version: {torch.__version__}\\ntorchvision version: {torchvision.__version__}\"), device"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def train_step(model: torch.nn.Module,\n",
        "               data_loader: torch.utils.data.DataLoader,\n",
        "               loss_fn: torch.nn.Module,\n",
        "               optimizer: torch.optim.Optimizer,\n",
        "               accuracy_fn,\n",
        "               device: torch.device = device):\n",
        "    train_loss, train_acc = 0, 0\n",
        "    model.to(device)\n",
        "    for batch, (X, y) in enumerate(data_loader):\n",
        "        # Send data to GPU\n",
        "        X, y = X.to(device), y.to(device)\n",
        "\n",
        "        # 1. Forward pass\n",
        "        y_pred = model(X)\n",
        "\n",
        "        # 2. Calculate loss\n",
        "        loss = loss_fn(y_pred, y)\n",
        "        train_loss += loss\n",
        "        train_acc += accuracy_fn(y_true=y,\n",
        "                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels\n",
        "\n",
        "        # 3. Optimizer zero grad\n",
        "        optimizer.zero_grad()\n",
        "\n",
        "        # 4. Loss backward\n",
        "        loss.backward()\n",
        "\n",
        "        # 5. Optimizer step\n",
        "        optimizer.step()\n",
        "\n",
        "    # Calculate loss and accuracy per epoch and print out what's happening\n",
        "    train_loss /= len(data_loader)\n",
        "    train_acc /= len(data_loader)\n",
        "    print(f\"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%\")\n",
        "\n",
        "def test_step(data_loader: torch.utils.data.DataLoader,\n",
        "              model: torch.nn.Module,\n",
        "              loss_fn: torch.nn.Module,\n",
        "              accuracy_fn,\n",
        "              device: torch.device = device):\n",
        "    test_loss, test_acc = 0, 0\n",
        "    model.to(device)\n",
        "    model.eval() # put model in eval mode\n",
        "    # Turn on inference context manager\n",
        "    with torch.inference_mode():\n",
        "        for X, y in data_loader:\n",
        "            # Send data to GPU\n",
        "            X, y = X.to(device), y.to(device)\n",
        "\n",
        "            # 1. Forward pass\n",
        "            test_pred = model(X)\n",
        "\n",
        "            # 2. Calculate loss and accuracy\n",
        "            test_loss += loss_fn(test_pred, y)\n",
        "            test_acc += accuracy_fn(y_true=y,\n",
        "                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels\n",
        "            )\n",
        "\n",
        "        # Adjust metrics and print out\n",
        "        test_loss /= len(data_loader)\n",
        "        test_acc /= len(data_loader)\n",
        "        print(f\"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\\n\")"
      ],
      "metadata": {
        "id": "9k1-LDQen_Mc"
      },
      "execution_count": 2,
      "outputs": []
    }
  ]
}