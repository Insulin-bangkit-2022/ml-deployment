{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copy of Capstone Deep Neural Network",
      "provenance": [],
      "collapsed_sections": [],
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
        "<a href=\"https://colab.research.google.com/github/Insulin-bangkit-2022/ml-deployment/blob/main/model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**IMPORT LIBRARIES**"
      ],
      "metadata": {
        "id": "TU94yADPye0c"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "s5SSMF1qjq_E"
      },
      "outputs": [],
      "source": [
        "import pandas as pd \n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import tensorflow as tf\n",
        "import numpy as np\n",
        "import os\n",
        "import pickle\n",
        "from google.colab import drive\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import confusion_matrix,accuracy_score\n",
        "from keras.models import load_model\n",
        "%matplotlib inline"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Import Diabetes Dataset from Kaggle**"
      ],
      "metadata": {
        "id": "fNuGEZI4yla5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "! pip install kaggle"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "O-PwhucxRAfh",
        "outputId": "b5bb26f7-c9b4-4180-abf1-cfecbf299ab9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Requirement already satisfied: kaggle in /usr/local/lib/python3.7/dist-packages (1.5.12)\n",
            "Requirement already satisfied: python-slugify in /usr/local/lib/python3.7/dist-packages (from kaggle) (6.1.2)\n",
            "Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.15.0)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from kaggle) (4.64.0)\n",
            "Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.24.3)\n",
            "Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from kaggle) (2022.5.18.1)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.23.0)\n",
            "Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.8.2)\n",
            "Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.7/dist-packages (from python-slugify->kaggle) (1.3)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (2.10)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (3.0.4)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "drive.mount('/content/gdrive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cvaxxJp8RHLD",
        "outputId": "561248ea-0765-4722-a926-f9bd1550fb6a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/gdrive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "os.environ['KAGGLE_CONFIG_DIR'] = \"/content/gdrive/Shared drives/Capstone Project/Product Based\""
      ],
      "metadata": {
        "id": "tIJwJrQZRNk4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#changing the working directory\n",
        "%cd /content/gdrive/Shared drives/Capstone Project/Product Based"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HqSjaknDUiNL",
        "outputId": "7750690c-e113-4bde-83d3-0035b5235c99"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/gdrive/Shared drives/Capstone Project/Product Based\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!kaggle datasets download -d andrewmvd/early-diabetes-classification"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aEofpxIGRQfQ",
        "outputId": "a504a57d-c20c-4414-c041-b41a9d0ce2b6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "early-diabetes-classification.zip: Skipping, found more recently modified local copy (use --force to force download)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Read Diabetes Dataset**"
      ],
      "metadata": {
        "id": "ngKxU0-qyxhA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data = pd.read_csv('diabetes_data.csv', delimiter = ';')\n",
        "data"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 488
        },
        "id": "bOom_CKIkHxv",
        "outputId": "71eebf65-1d35-4687-c022-491776b85841"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     age  gender  polyuria  polydipsia  sudden_weight_loss  weakness  \\\n",
              "0     40    Male         0           1                   0         1   \n",
              "1     58    Male         0           0                   0         1   \n",
              "2     41    Male         1           0                   0         1   \n",
              "3     45    Male         0           0                   1         1   \n",
              "4     60    Male         1           1                   1         1   \n",
              "..   ...     ...       ...         ...                 ...       ...   \n",
              "515   39  Female         1           1                   1         0   \n",
              "516   48  Female         1           1                   1         1   \n",
              "517   58  Female         1           1                   1         1   \n",
              "518   32  Female         0           0                   0         1   \n",
              "519   42    Male         0           0                   0         0   \n",
              "\n",
              "     polyphagia  genital_thrush  visual_blurring  itching  irritability  \\\n",
              "0             0               0                0        1             0   \n",
              "1             0               0                1        0             0   \n",
              "2             1               0                0        1             0   \n",
              "3             1               1                0        1             0   \n",
              "4             1               0                1        1             1   \n",
              "..          ...             ...              ...      ...           ...   \n",
              "515           1               0                0        1             0   \n",
              "516           1               0                0        1             1   \n",
              "517           1               0                1        0             0   \n",
              "518           0               0                1        1             0   \n",
              "519           0               0                0        0             0   \n",
              "\n",
              "     delayed_healing  partial_paresis  muscle_stiffness  alopecia  obesity  \\\n",
              "0                  1                0                 1         1        1   \n",
              "1                  0                1                 0         1        0   \n",
              "2                  1                0                 1         1        0   \n",
              "3                  1                0                 0         0        0   \n",
              "4                  1                1                 1         1        1   \n",
              "..               ...              ...               ...       ...      ...   \n",
              "515                1                1                 0         0        0   \n",
              "516                1                1                 0         0        0   \n",
              "517                0                1                 1         0        1   \n",
              "518                1                0                 0         1        0   \n",
              "519                0                0                 0         0        0   \n",
              "\n",
              "     class  \n",
              "0        1  \n",
              "1        1  \n",
              "2        1  \n",
              "3        1  \n",
              "4        1  \n",
              "..     ...  \n",
              "515      1  \n",
              "516      1  \n",
              "517      1  \n",
              "518      0  \n",
              "519      0  \n",
              "\n",
              "[520 rows x 17 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-5b8f7038-15bb-402d-aab5-ae94c5369ffe\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>gender</th>\n",
              "      <th>polyuria</th>\n",
              "      <th>polydipsia</th>\n",
              "      <th>sudden_weight_loss</th>\n",
              "      <th>weakness</th>\n",
              "      <th>polyphagia</th>\n",
              "      <th>genital_thrush</th>\n",
              "      <th>visual_blurring</th>\n",
              "      <th>itching</th>\n",
              "      <th>irritability</th>\n",
              "      <th>delayed_healing</th>\n",
              "      <th>partial_paresis</th>\n",
              "      <th>muscle_stiffness</th>\n",
              "      <th>alopecia</th>\n",
              "      <th>obesity</th>\n",
              "      <th>class</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>40</td>\n",
              "      <td>Male</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>58</td>\n",
              "      <td>Male</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>41</td>\n",
              "      <td>Male</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>45</td>\n",
              "      <td>Male</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>60</td>\n",
              "      <td>Male</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>515</th>\n",
              "      <td>39</td>\n",
              "      <td>Female</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>516</th>\n",
              "      <td>48</td>\n",
              "      <td>Female</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>517</th>\n",
              "      <td>58</td>\n",
              "      <td>Female</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>518</th>\n",
              "      <td>32</td>\n",
              "      <td>Female</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>519</th>\n",
              "      <td>42</td>\n",
              "      <td>Male</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>520 rows × 17 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-5b8f7038-15bb-402d-aab5-ae94c5369ffe')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-5b8f7038-15bb-402d-aab5-ae94c5369ffe button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-5b8f7038-15bb-402d-aab5-ae94c5369ffe');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data[\"gender\"] = data[\"gender\"].apply({\"Male\":1, \"Female\":0}.get)\n",
        "data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 270
        },
        "id": "gRJpDQLDmPuh",
        "outputId": "1bd7cdbe-8e8f-4752-deb1-86767cb332cb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   age  gender  polyuria  polydipsia  sudden_weight_loss  weakness  \\\n",
              "0   40       1         0           1                   0         1   \n",
              "1   58       1         0           0                   0         1   \n",
              "2   41       1         1           0                   0         1   \n",
              "3   45       1         0           0                   1         1   \n",
              "4   60       1         1           1                   1         1   \n",
              "\n",
              "   polyphagia  genital_thrush  visual_blurring  itching  irritability  \\\n",
              "0           0               0                0        1             0   \n",
              "1           0               0                1        0             0   \n",
              "2           1               0                0        1             0   \n",
              "3           1               1                0        1             0   \n",
              "4           1               0                1        1             1   \n",
              "\n",
              "   delayed_healing  partial_paresis  muscle_stiffness  alopecia  obesity  \\\n",
              "0                1                0                 1         1        1   \n",
              "1                0                1                 0         1        0   \n",
              "2                1                0                 1         1        0   \n",
              "3                1                0                 0         0        0   \n",
              "4                1                1                 1         1        1   \n",
              "\n",
              "   class  \n",
              "0      1  \n",
              "1      1  \n",
              "2      1  \n",
              "3      1  \n",
              "4      1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-6ed2680f-4ee3-4840-8cb2-8f9479c523e8\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>gender</th>\n",
              "      <th>polyuria</th>\n",
              "      <th>polydipsia</th>\n",
              "      <th>sudden_weight_loss</th>\n",
              "      <th>weakness</th>\n",
              "      <th>polyphagia</th>\n",
              "      <th>genital_thrush</th>\n",
              "      <th>visual_blurring</th>\n",
              "      <th>itching</th>\n",
              "      <th>irritability</th>\n",
              "      <th>delayed_healing</th>\n",
              "      <th>partial_paresis</th>\n",
              "      <th>muscle_stiffness</th>\n",
              "      <th>alopecia</th>\n",
              "      <th>obesity</th>\n",
              "      <th>class</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>40</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>58</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>41</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>45</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>60</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-6ed2680f-4ee3-4840-8cb2-8f9479c523e8')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-6ed2680f-4ee3-4840-8cb2-8f9479c523e8 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-6ed2680f-4ee3-4840-8cb2-8f9479c523e8');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Plotting Heat Map"
      ],
      "metadata": {
        "id": "AC25zIaIy4nh"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "tc = data.corr()\n",
        "sns.heatmap(tc,annot = False,cmap=\"coolwarm\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 377
        },
        "id": "mBSAwOYFkMKw",
        "outputId": "b5fc58a5-e6b2-49f1-a88a-34c360e3a206"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f2b998fee50>"
            ]
          },
          "metadata": {},
          "execution_count": 9
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LMzXXfGRmFEB",
        "outputId": "c1d84794-63a6-4572-87c8-30b4958a6f61"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 520 entries, 0 to 519\n",
            "Data columns (total 17 columns):\n",
            " #   Column              Non-Null Count  Dtype\n",
            "---  ------              --------------  -----\n",
            " 0   age                 520 non-null    int64\n",
            " 1   gender              520 non-null    int64\n",
            " 2   polyuria            520 non-null    int64\n",
            " 3   polydipsia          520 non-null    int64\n",
            " 4   sudden_weight_loss  520 non-null    int64\n",
            " 5   weakness            520 non-null    int64\n",
            " 6   polyphagia          520 non-null    int64\n",
            " 7   genital_thrush      520 non-null    int64\n",
            " 8   visual_blurring     520 non-null    int64\n",
            " 9   itching             520 non-null    int64\n",
            " 10  irritability        520 non-null    int64\n",
            " 11  delayed_healing     520 non-null    int64\n",
            " 12  partial_paresis     520 non-null    int64\n",
            " 13  muscle_stiffness    520 non-null    int64\n",
            " 14  alopecia            520 non-null    int64\n",
            " 15  obesity             520 non-null    int64\n",
            " 16  class               520 non-null    int64\n",
            "dtypes: int64(17)\n",
            "memory usage: 69.2 KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Split the Dataset Into Training and Test Set**"
      ],
      "metadata": {
        "id": "KfBjL3zuy-UP"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x = data[data.columns[:-1]]\n",
        "y = data[data.columns[-1]]\n",
        "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.15,random_state = 10)\n",
        "print(x_train)\n",
        "print(y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HpAUhdwomk86",
        "outputId": "2e759c53-0d5e-4258-c7f0-34deedd6063b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     age  gender  polyuria  polydipsia  sudden_weight_loss  weakness  \\\n",
            "260   33       1         0           1                   0         0   \n",
            "184   48       0         1           1                   1         0   \n",
            "172   60       1         1           0                   0         0   \n",
            "193   36       1         1           0                   0         1   \n",
            "154   62       1         1           0                   0         1   \n",
            "..   ...     ...       ...         ...                 ...       ...   \n",
            "123   47       1         0           1                   0         0   \n",
            "369   47       1         0           0                   0         0   \n",
            "320   65       0         0           0                   0         1   \n",
            "125   55       1         1           1                   1         1   \n",
            "265   28       0         0           0                   0         0   \n",
            "\n",
            "     polyphagia  genital_thrush  visual_blurring  itching  irritability  \\\n",
            "260           0               0                0        0             0   \n",
            "184           1               1                0        0             1   \n",
            "172           0               0                1        0             0   \n",
            "193           0               1                1        1             0   \n",
            "154           0               1                1        1             1   \n",
            "..          ...             ...              ...      ...           ...   \n",
            "123           0               0                1        1             0   \n",
            "369           0               0                0        1             0   \n",
            "320           0               0                0        1             0   \n",
            "125           1               0                1        0             0   \n",
            "265           0               0                1        0             0   \n",
            "\n",
            "     delayed_healing  partial_paresis  muscle_stiffness  alopecia  obesity  \n",
            "260                0                0                 0         0        0  \n",
            "184                1                0                 1         1        1  \n",
            "172                0                1                 0         1        0  \n",
            "193                1                0                 0         0        0  \n",
            "154                0                1                 1         1        1  \n",
            "..               ...              ...               ...       ...      ...  \n",
            "123                0                0                 0         1        1  \n",
            "369                0                0                 0         1        0  \n",
            "320                1                0                 0         1        0  \n",
            "125                1                1                 0         0        0  \n",
            "265                0                1                 1         0        0  \n",
            "\n",
            "[442 rows x 16 columns]\n",
            "260    1\n",
            "184    1\n",
            "172    1\n",
            "193    1\n",
            "154    1\n",
            "      ..\n",
            "123    1\n",
            "369    0\n",
            "320    0\n",
            "125    1\n",
            "265    1\n",
            "Name: class, Length: 442, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Training Dataset**"
      ],
      "metadata": {
        "id": "V0e7qc-4zHIl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = tf.keras.models.Sequential([\n",
        "  tf.keras.layers.Dense(units = 128, activation = \"relu\"),\n",
        "  tf.keras.layers.Dropout(0.2),\n",
        "  tf.keras.layers.Dense(units = 32, activation = \"relu\"),\n",
        "  tf.keras.layers.Dropout(0.1),\n",
        "  tf.keras.layers.Dense(units = 1,activation = \"sigmoid\")\n",
        "  ])\n",
        "\n",
        "model.compile(optimizer = \"adam\", \n",
        "              loss = \"binary_crossentropy\" , \n",
        "              metrics=[\"accuracy\"])\n",
        "\n",
        "#Here we train our model.\n",
        "history = model.fit(x_train,y_train,epochs = 100,validation_data = (x_test,y_test))\n",
        "#This the inference phase.We try our model on test data.\n",
        "y_pred = model.predict(x_test)\n",
        "y_pred = (y_pred > 0.5)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "95Tdsk7onrug",
        "outputId": "2b6a696a-1a23-449e-ed64-58c722dc7cf9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/100\n",
            "14/14 [==============================] - 2s 17ms/step - loss: 1.9873 - accuracy: 0.4638 - val_loss: 0.7260 - val_accuracy: 0.6026\n",
            "Epoch 2/100\n",
            "14/14 [==============================] - 0s 13ms/step - loss: 1.4833 - accuracy: 0.5633 - val_loss: 0.6586 - val_accuracy: 0.6026\n",
            "Epoch 3/100\n",
            "14/14 [==============================] - 0s 13ms/step - loss: 1.1498 - accuracy: 0.5747 - val_loss: 0.5563 - val_accuracy: 0.8077\n",
            "Epoch 4/100\n",
            "14/14 [==============================] - 0s 14ms/step - loss: 1.0280 - accuracy: 0.5543 - val_loss: 0.7423 - val_accuracy: 0.6026\n",
            "Epoch 5/100\n",
            "14/14 [==============================] - 0s 13ms/step - loss: 0.8263 - accuracy: 0.5769 - val_loss: 0.6745 - val_accuracy: 0.6026\n",
            "Epoch 6/100\n",
            "14/14 [==============================] - 0s 10ms/step - loss: 0.8026 - accuracy: 0.6131 - val_loss: 0.6500 - val_accuracy: 0.6026\n",
            "Epoch 7/100\n",
            "14/14 [==============================] - 0s 14ms/step - loss: 0.7812 - accuracy: 0.5769 - val_loss: 0.6650 - val_accuracy: 0.6026\n",
            "Epoch 8/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.6895 - accuracy: 0.6312 - val_loss: 0.6367 - val_accuracy: 0.6026\n",
            "Epoch 9/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.7127 - accuracy: 0.5950 - val_loss: 0.6488 - val_accuracy: 0.6026\n",
            "Epoch 10/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.7221 - accuracy: 0.5837 - val_loss: 0.6235 - val_accuracy: 0.6026\n",
            "Epoch 11/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.6392 - accuracy: 0.6719 - val_loss: 0.6176 - val_accuracy: 0.7564\n",
            "Epoch 12/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.6323 - accuracy: 0.6561 - val_loss: 0.5915 - val_accuracy: 0.6026\n",
            "Epoch 13/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.6384 - accuracy: 0.6312 - val_loss: 0.5878 - val_accuracy: 0.7821\n",
            "Epoch 14/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.6150 - accuracy: 0.6538 - val_loss: 0.5639 - val_accuracy: 0.7179\n",
            "Epoch 15/100\n",
            "14/14 [==============================] - 0s 6ms/step - loss: 0.5869 - accuracy: 0.6742 - val_loss: 0.5096 - val_accuracy: 0.7821\n",
            "Epoch 16/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.5959 - accuracy: 0.6765 - val_loss: 0.5301 - val_accuracy: 0.8333\n",
            "Epoch 17/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.5643 - accuracy: 0.6787 - val_loss: 0.4577 - val_accuracy: 0.7564\n",
            "Epoch 18/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.5383 - accuracy: 0.7036 - val_loss: 0.4230 - val_accuracy: 0.8846\n",
            "Epoch 19/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.5491 - accuracy: 0.7014 - val_loss: 0.3970 - val_accuracy: 0.8718\n",
            "Epoch 20/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.4849 - accuracy: 0.7760 - val_loss: 0.3653 - val_accuracy: 0.8718\n",
            "Epoch 21/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.4914 - accuracy: 0.7557 - val_loss: 0.3771 - val_accuracy: 0.9103\n",
            "Epoch 22/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.4657 - accuracy: 0.7760 - val_loss: 0.3275 - val_accuracy: 0.8718\n",
            "Epoch 23/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.4225 - accuracy: 0.7964 - val_loss: 0.2905 - val_accuracy: 0.9103\n",
            "Epoch 24/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.4461 - accuracy: 0.7783 - val_loss: 0.2700 - val_accuracy: 0.9103\n",
            "Epoch 25/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.3786 - accuracy: 0.8416 - val_loss: 0.2607 - val_accuracy: 0.9359\n",
            "Epoch 26/100\n",
            "14/14 [==============================] - 0s 6ms/step - loss: 0.3798 - accuracy: 0.8439 - val_loss: 0.2471 - val_accuracy: 0.8846\n",
            "Epoch 27/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.3868 - accuracy: 0.8348 - val_loss: 0.2405 - val_accuracy: 0.8846\n",
            "Epoch 28/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.3330 - accuracy: 0.8462 - val_loss: 0.2239 - val_accuracy: 0.8846\n",
            "Epoch 29/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.3540 - accuracy: 0.8394 - val_loss: 0.2124 - val_accuracy: 0.8846\n",
            "Epoch 30/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.3480 - accuracy: 0.8281 - val_loss: 0.2008 - val_accuracy: 0.9103\n",
            "Epoch 31/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.3299 - accuracy: 0.8416 - val_loss: 0.2042 - val_accuracy: 0.8974\n",
            "Epoch 32/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.3334 - accuracy: 0.8462 - val_loss: 0.2054 - val_accuracy: 0.8846\n",
            "Epoch 33/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2973 - accuracy: 0.8620 - val_loss: 0.2012 - val_accuracy: 0.8846\n",
            "Epoch 34/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.3124 - accuracy: 0.8665 - val_loss: 0.2022 - val_accuracy: 0.8846\n",
            "Epoch 35/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.3169 - accuracy: 0.8665 - val_loss: 0.1945 - val_accuracy: 0.9231\n",
            "Epoch 36/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2970 - accuracy: 0.8710 - val_loss: 0.2108 - val_accuracy: 0.8974\n",
            "Epoch 37/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2869 - accuracy: 0.8756 - val_loss: 0.1800 - val_accuracy: 0.8974\n",
            "Epoch 38/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2827 - accuracy: 0.8710 - val_loss: 0.1767 - val_accuracy: 0.8974\n",
            "Epoch 39/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2628 - accuracy: 0.8824 - val_loss: 0.2017 - val_accuracy: 0.8974\n",
            "Epoch 40/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.3022 - accuracy: 0.8778 - val_loss: 0.1763 - val_accuracy: 0.8974\n",
            "Epoch 41/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2498 - accuracy: 0.8801 - val_loss: 0.1756 - val_accuracy: 0.8974\n",
            "Epoch 42/100\n",
            "14/14 [==============================] - 0s 6ms/step - loss: 0.2899 - accuracy: 0.8756 - val_loss: 0.1903 - val_accuracy: 0.9103\n",
            "Epoch 43/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2756 - accuracy: 0.8869 - val_loss: 0.1902 - val_accuracy: 0.9103\n",
            "Epoch 44/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2501 - accuracy: 0.8914 - val_loss: 0.1896 - val_accuracy: 0.9103\n",
            "Epoch 45/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2569 - accuracy: 0.8937 - val_loss: 0.1860 - val_accuracy: 0.9103\n",
            "Epoch 46/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2698 - accuracy: 0.8937 - val_loss: 0.1857 - val_accuracy: 0.9103\n",
            "Epoch 47/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2384 - accuracy: 0.8959 - val_loss: 0.1666 - val_accuracy: 0.8974\n",
            "Epoch 48/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2278 - accuracy: 0.9208 - val_loss: 0.1764 - val_accuracy: 0.9103\n",
            "Epoch 49/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2414 - accuracy: 0.9095 - val_loss: 0.1792 - val_accuracy: 0.9103\n",
            "Epoch 50/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2162 - accuracy: 0.9095 - val_loss: 0.1707 - val_accuracy: 0.9103\n",
            "Epoch 51/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2383 - accuracy: 0.8937 - val_loss: 0.1788 - val_accuracy: 0.9103\n",
            "Epoch 52/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2283 - accuracy: 0.9118 - val_loss: 0.1784 - val_accuracy: 0.9103\n",
            "Epoch 53/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2372 - accuracy: 0.9118 - val_loss: 0.1619 - val_accuracy: 0.9359\n",
            "Epoch 54/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2413 - accuracy: 0.8982 - val_loss: 0.1796 - val_accuracy: 0.9103\n",
            "Epoch 55/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2350 - accuracy: 0.9140 - val_loss: 0.1714 - val_accuracy: 0.9103\n",
            "Epoch 56/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2555 - accuracy: 0.8937 - val_loss: 0.1533 - val_accuracy: 0.9487\n",
            "Epoch 57/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2346 - accuracy: 0.9118 - val_loss: 0.1649 - val_accuracy: 0.9359\n",
            "Epoch 58/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2259 - accuracy: 0.9027 - val_loss: 0.1684 - val_accuracy: 0.9103\n",
            "Epoch 59/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2137 - accuracy: 0.9208 - val_loss: 0.1541 - val_accuracy: 0.9359\n",
            "Epoch 60/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2204 - accuracy: 0.9163 - val_loss: 0.1848 - val_accuracy: 0.9103\n",
            "Epoch 61/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2497 - accuracy: 0.8869 - val_loss: 0.1542 - val_accuracy: 0.9103\n",
            "Epoch 62/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2225 - accuracy: 0.9118 - val_loss: 0.1576 - val_accuracy: 0.9103\n",
            "Epoch 63/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2107 - accuracy: 0.9095 - val_loss: 0.1923 - val_accuracy: 0.9103\n",
            "Epoch 64/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2318 - accuracy: 0.9095 - val_loss: 0.1598 - val_accuracy: 0.9103\n",
            "Epoch 65/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2062 - accuracy: 0.9072 - val_loss: 0.1509 - val_accuracy: 0.9359\n",
            "Epoch 66/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2168 - accuracy: 0.9208 - val_loss: 0.1771 - val_accuracy: 0.9103\n",
            "Epoch 67/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2200 - accuracy: 0.9208 - val_loss: 0.1408 - val_accuracy: 0.9487\n",
            "Epoch 68/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2159 - accuracy: 0.9027 - val_loss: 0.1519 - val_accuracy: 0.9359\n",
            "Epoch 69/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2160 - accuracy: 0.9208 - val_loss: 0.1889 - val_accuracy: 0.9103\n",
            "Epoch 70/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2373 - accuracy: 0.9072 - val_loss: 0.1447 - val_accuracy: 0.9359\n",
            "Epoch 71/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2389 - accuracy: 0.8982 - val_loss: 0.1568 - val_accuracy: 0.9359\n",
            "Epoch 72/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2127 - accuracy: 0.9095 - val_loss: 0.1475 - val_accuracy: 0.9359\n",
            "Epoch 73/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2175 - accuracy: 0.9231 - val_loss: 0.1621 - val_accuracy: 0.9103\n",
            "Epoch 74/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2143 - accuracy: 0.9276 - val_loss: 0.1458 - val_accuracy: 0.9359\n",
            "Epoch 75/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.1981 - accuracy: 0.9276 - val_loss: 0.1582 - val_accuracy: 0.9359\n",
            "Epoch 76/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.1934 - accuracy: 0.9231 - val_loss: 0.1555 - val_accuracy: 0.9359\n",
            "Epoch 77/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.1933 - accuracy: 0.9253 - val_loss: 0.1608 - val_accuracy: 0.9103\n",
            "Epoch 78/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2054 - accuracy: 0.9253 - val_loss: 0.1494 - val_accuracy: 0.9359\n",
            "Epoch 79/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2118 - accuracy: 0.9118 - val_loss: 0.1761 - val_accuracy: 0.9103\n",
            "Epoch 80/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2156 - accuracy: 0.9072 - val_loss: 0.1526 - val_accuracy: 0.9359\n",
            "Epoch 81/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2073 - accuracy: 0.9208 - val_loss: 0.1551 - val_accuracy: 0.9359\n",
            "Epoch 82/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2234 - accuracy: 0.9072 - val_loss: 0.1770 - val_accuracy: 0.9103\n",
            "Epoch 83/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.2136 - accuracy: 0.9208 - val_loss: 0.1488 - val_accuracy: 0.9359\n",
            "Epoch 84/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2096 - accuracy: 0.9163 - val_loss: 0.1536 - val_accuracy: 0.9359\n",
            "Epoch 85/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.2064 - accuracy: 0.9186 - val_loss: 0.1561 - val_accuracy: 0.9359\n",
            "Epoch 86/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.1887 - accuracy: 0.9321 - val_loss: 0.1332 - val_accuracy: 0.9487\n",
            "Epoch 87/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2046 - accuracy: 0.9208 - val_loss: 0.1634 - val_accuracy: 0.9103\n",
            "Epoch 88/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.1995 - accuracy: 0.9299 - val_loss: 0.1451 - val_accuracy: 0.9359\n",
            "Epoch 89/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.1953 - accuracy: 0.9276 - val_loss: 0.1335 - val_accuracy: 0.9487\n",
            "Epoch 90/100\n",
            "14/14 [==============================] - 0s 3ms/step - loss: 0.1954 - accuracy: 0.9253 - val_loss: 0.1578 - val_accuracy: 0.9103\n",
            "Epoch 91/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2067 - accuracy: 0.9140 - val_loss: 0.1539 - val_accuracy: 0.9103\n",
            "Epoch 92/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.1992 - accuracy: 0.9208 - val_loss: 0.1498 - val_accuracy: 0.9359\n",
            "Epoch 93/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2086 - accuracy: 0.9163 - val_loss: 0.1990 - val_accuracy: 0.9103\n",
            "Epoch 94/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.1969 - accuracy: 0.9118 - val_loss: 0.1523 - val_accuracy: 0.9359\n",
            "Epoch 95/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.1807 - accuracy: 0.9299 - val_loss: 0.1462 - val_accuracy: 0.9359\n",
            "Epoch 96/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2061 - accuracy: 0.9253 - val_loss: 0.2016 - val_accuracy: 0.8974\n",
            "Epoch 97/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.2005 - accuracy: 0.9186 - val_loss: 0.1392 - val_accuracy: 0.9487\n",
            "Epoch 98/100\n",
            "14/14 [==============================] - 0s 5ms/step - loss: 0.1995 - accuracy: 0.9276 - val_loss: 0.1323 - val_accuracy: 0.9487\n",
            "Epoch 99/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.1816 - accuracy: 0.9253 - val_loss: 0.1581 - val_accuracy: 0.9359\n",
            "Epoch 100/100\n",
            "14/14 [==============================] - 0s 4ms/step - loss: 0.1763 - accuracy: 0.9276 - val_loss: 0.1682 - val_accuracy: 0.9359\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Plot Accuracy and Loss**"
      ],
      "metadata": {
        "id": "QhUkzure3hRS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#-----------------------------------------------------------\n",
        "# Retrieve a list of list results on training and test data\n",
        "# sets for each training epoch\n",
        "#-----------------------------------------------------------\n",
        "acc      = history.history[     'accuracy' ]\n",
        "val_acc  = history.history[ 'val_accuracy' ]\n",
        "loss     = history.history[    'loss' ]\n",
        "val_loss = history.history['val_loss' ]\n",
        "\n",
        "epochs   = range(len(acc)) # Get number of epochs\n",
        "\n",
        "#------------------------------------------------\n",
        "# Plot training and validation accuracy per epoch\n",
        "#------------------------------------------------\n",
        "plt.plot  ( epochs,     acc )\n",
        "plt.plot  ( epochs, val_acc )\n",
        "plt.title ('Training and validation accuracy')\n",
        "plt.figure()\n",
        "\n",
        "#------------------------------------------------\n",
        "# Plot training and validation loss per epoch\n",
        "#------------------------------------------------\n",
        "plt.plot  ( epochs,     loss )\n",
        "plt.plot  ( epochs, val_loss )\n",
        "plt.title ('Training and validation loss'   )"
      ],
      "metadata": {
        "id": "_hwSfw-50GEu",
        "outputId": "7a7bf64e-b828-498f-f572-2a161ef1ed45",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 563
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Training and validation loss')"
            ]
          },
          "metadata": {},
          "execution_count": 13
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAEICAYAAABPgw/pAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOydeXhdVdX/PyvznLRJ06ZJ57nQlo7MUGYKCFJQJlFARQXU1xcUeEXgh6KiKK++oojKPMtkpSOlLcjcUlpK57R0SJp5npObu39/7HNzh9w0Q2+a3tv1eZ4895x99jlnnXNuvnedtdfeW4wxKIqiKOFP1EAboCiKooQGFXRFUZQIQQVdURQlQlBBVxRFiRBU0BVFUSIEFXRFUZQIQQU9ghGRpSLyjVDXHUhEZI+InN0PxzUiMt5ZfkREftaTun04zzUisqKvdirKwRDNQz+yEJF6n9UkoAVod9a/Y4x59vBbdeQgInuAbxljVob4uAaYYIzJD1VdERkNfAHEGmNcobBTUQ5GzEAboPhjjEnxLB9MvEQkRkVCOVLQ7+ORgYZcwgQRmS8iBSJyu4gUA4+LyCAReUNEykSkylnO89lnjYh8y1m+TkTeFZEHnbpfiMiCPtYdIyLviEidiKwUkYdF5Jku7O6JjT8Xkfec460QkSyf7deKyF4RqRCRnx7k/hwvIsUiEu1TdqmIfOYszxORD0SkWkSKRORPIhLXxbGeEJFf+Kz/2NnngIjcEFD3QhH5VERqRWS/iNzrs/kd57NaROpF5ETPvfXZ/yQRWSsiNc7nST29N728z4NF5HHnGqpE5HWfbZeIyAbnGnaJyPlOuV94S0Tu9TxnERnthJ6+KSL7gFVO+T+d51DjfEeO8dk/UUR+5zzPGuc7ligii0Xk+wHX85mIXBrsWpWuUUEPL4YBg4FRwI3Y5/e4sz4SaAL+dJD9jwe2A1nAb4B/iIj0oe5zwMdAJnAvcO1BztkTG68GrgeygTjgNgARmQr8xTn+cOd8eQTBGPMR0ACcGXDc55zlduBHzvWcCJwF3HQQu3FsON+x5xxgAhAYv28Avg5kABcC3xORLzvbTnM+M4wxKcaYDwKOPRhYDPzRubbfA4tFJDPgGjrdmyB0d5+fxobwjnGO9ZBjwzzgKeDHzjWcBuzp6n4E4XRgCnCes74Ue5+ygfWAb4jwQWA2cBL2e/wTwA08CXzNU0lEZgC52Huj9AZjjP4doX/Yf6yzneX5QCuQcJD6xwFVPutrsCEbgOuAfJ9tSYABhvWmLlYsXECSz/ZngGd6eE3BbLzLZ/0mYJmzfDfwgs+2ZOcenN3FsX8BPOYsp2LFdlQXdf8LeM1n3QDjneUngF84y48Bv/apN9G3bpDj/i/wkLM82qkb47P9OuBdZ/la4OOA/T8Aruvu3vTmPgM5WOEcFKTeXz32Huz756zf63nOPtc29iA2ZDh10rE/OE3AjCD1EoAqbLsEWOH/8+H+f4uEP/XQw4syY0yzZ0VEkkTkr84rbC32FT/DN+wQQLFnwRjT6Cym9LLucKDSpwxgf1cG99DGYp/lRh+bhvse2xjTAFR0dS6sN75QROKBhcB6Y8xex46JThii2LHjl1hvvTv8bAD2Blzf8SKy2gl11ADf7eFxPcfeG1C2F+udeujq3vjRzX0egX1mVUF2HQHs6qG9wei4NyISLSK/dsI2tXg9/SznLyHYuZzv9IvA10QkCrgK+0ah9BIV9PAiMCXpVmAScLwxJg3vK35XYZRQUAQMFpEkn7IRB6l/KDYW+R7bOWdmV5WNMVuwgrgA/3AL2NDNNqwXmAb8T19swL6h+PIcsAgYYYxJBx7xOW53KWQHsCESX0YChT2wK5CD3ef92GeWEWS//cC4Lo7ZgH078zAsSB3fa7wauAQblkrHevEeG8qB5oOc60ngGmworNEEhKeUnqGCHt6kYl9jq5147D39fULH410H3CsicSJyIvClfrLxZeAiETnFacC8j+6/s88BP8QK2j8D7KgF6kVkMvC9HtrwEnCdiEx1flAC7U/Fer/NTjz6ap9tZdhQx9gujr0EmCgiV4tIjIhcAUwF3uihbYF2BL3PxpgibGz7z07jaayIeAT/H8D1InKWiESJSK5zfwA2AFc69ecAl/fAhhbsW1QS9i3IY4MbG776vYgMd7z5E523KRwBdwO/Q73zPqOCHt78L5CI9X4+BJYdpvNeg21YrMDGrV/E/iMHo882GmM2AzdjRboIG2ct6Ga357ENdauMMeU+5bdhxbYO+Jtjc09sWOpcwyog3/n05SbgPhGpw8b8X/LZtxG4H3hPbHbNCQHHrgAuwnrXFdhGwosC7O4p3d3na4E27FtKKbYNAWPMx9hG14eAGuBtvG8NP8N61FXA/8P/jScYT2HfkAqBLY4dvtwGbALWApXAA/hr0FPANGybjNIHtGORcsiIyIvANmNMv78hKJGLiHwduNEYc8pA2xKuqIeu9BoRmSsi45xX9POxcdPXu9tPUbrCCWfdBDw60LaEMyroSl8Yhk2pq8fmUH/PGPPpgFqkhC0ich62vaGE7sM6ykHQkIuiKEqEoB66oihKhDBgg3NlZWWZ0aNHD9TpFUVRwpJPPvmk3BgzJNi2ARP00aNHs27duoE6vaIoSlgiIoG9izvQkIuiKEqEoIKuKIoSIaigK4qiRAgq6IqiKBGCCrqiKEqEoIKuKIoSIaigK4qiRAgDloeuKBFLUzVs+RfM/BpEdTV51GGgcjcUb4KplwycDYcDdztseNZeZ0J61/W2LYasiZA1oX/saKmDj/4KLmck6cQMmHMDxCb2z/mCoIKuKKFm4wuw7HZwu2DuNwfOjv/8Dj59Bm5YASOPHzg7+ptPHofFt0JrI5zw3eB1DnwKL1wDxy6Eyx/rHzu2LYZVP3dWBDBQXwrn/L/+OV8QNOSiKKGmaIP9XPVzaKwcODsObLSfS39svdhIpLESVv3CLhdtDF7HGFjyE8B0XScUVOSDRMFdZXBvNcy4Gj54GCoOZcrW3qGCriihpmijfbVvroHV9w+MDW3NULYVsqdae9Y/NTB29DerfgHNtfZ+dyXWn70IBR/be1GRb+v3BxX5kDEKYuLs+tn3QkwCLLujf84XBBV0RQklrY1Qtg2mfhnmfgvWPWbj2Ieb0i025HP6T2DUyfDWfdBUdfjt6E+KPrPhlrnfsvHzsm3Q1uRfp6UO3rwbcmfDWc6EWiWf9489Fbsgc7x3PXUozL8ddq6AHcv755wBaAw9UmmstI0zaTkDbUlk0VwLzdWQMTL49pLNYNyQMwNGnwybXrav+9cvARH/uqVbIWsSRB2iX9VYCe2tkDrMW+YJ++QcBwsegL+eBqt/CRf81n9fdzvsfQ9crZ2PGxNnfwwCG3aLP4e64uC25M2GxEH+ZQ3l9p6kZPuXN9dAwVrwTMmQngvZU/zrtLusfe1tnc/1zm/suc64E/a8C6bd3v+8Od46b/8G6kvgyuchPc+WHdgAo04Kbn9X/zct9bD/Q6+tqcNg2LHe7cZYQQ887rzv2LejpbfbcAzOd2DIJMgYEdyGQ0AFPVJ5/SaoLYDvvjvQlkQWy++E7Uvh1h0QHeTfp0NIZ1ixOfMuWPzfVpRG+0yVWbIZ/nISXPKwzYY5FF75FtQVwU0f+Nix0WZ8DBptf0hmXw9r/wGn3wHJmd56n78Kr36r62Mv/DtM/4p3vaEcHp0P7iACC3Ds5XD5P/zLXvq69ZxvXO1fvvyn8OnT3vXoePjxTv9MlY3PwaLvd23fxX+y9zlnhl0v2uAV9OZa+OgRG8vOm23LUoYdPI7+6o1Qsx9u/si/fNXP7bE8RMXCT3Z5ba0rhrYGfw8d7I/iggfg6YXw7OXe8gt/3y8N5irokUhrA+xaZb2idldw4VF6j7vdinljBez/yHrggRRthKRMrzc4/avWO9uxzF/Qty+xn9sWH5qgN9fCF2/b8ErFLsgc57UjZ4b3rWDmNbDuH5C/EmZc4WPHYkgZClc82/nYL15j7fQV9PyVVswX/t3+WPjy/h9h55vWm46OtWUNFbD3fcBA7QFIG27L3W57LyecB6f9GMq3w79uhl2r4Zgve4+5bQmkjwyemRKfCtmTqW9xcf9b1fw8YRAxvmK9e7V9c/G9v8OP61rQW+pg9xp7feX5kOWIszHWjjGnw5k/sxkzS39sQz5jTrV1Kp2GT8/992XcmfCD9fZeeBg0KrgNh4jG0COR3W9De4v9YtbsH2hrIofCT6yYgxXoYAQKaXyqFf4dK/zreWKqu9fYBsy+smuVFXOwsVqwglqy2eu1AuTMhORsf7vb2yD/LZhwLoyY2/lvwjl2u2+4Y8cy+wNw7GWd60//KrTUwL4PvfXz36QjTuEbRz6wHhrLYdpXnH2vhIQM7zWA9ep3r4FJ5we3L3syAI+s2cXza/ez0TUK9wGvWLdtXUpDVAq3fhjPG58doLa5zd6T8u22raPTvVztffPY6WNr6Vao2WdTHkfMZW3yaQDs3+LzRlSRbz8HBxF0gMFjO+w2eXNoTwo6P8Uho4Ieifh+GQ9jylTEs2M5SLSNS+9c0Xm7q8X+8/sKKcDE862IVH5h1xvKoWAd5M2FtkYb/+0rO1fY1/7M8V7BLNtmPdOc47z1oqKscO96y761gRXellqYeF7wY084zwr0fif80N5G246VLG+dzrxfrWLe/Ss5/pcrufPVz6hsaIWx820owvf7t2O5/SFJH+l/z3YstzHl8WfZ9egYGH+2reN227I974KrydrRBQeqm/jbf3YzeVgqHzePwJRsBlcr7vZ2mrYsZVXbNFbtqOSW5z5l1n1v8rtNCWDcFO1YG+ReLof4dMic4P/j47meCecC8Ov/VHLADGbdh2u47Z8bKa9vsYIeHd/xZrZhfzXnPvQ2P3pxA298doDy+hZWby/lrtc3cfKvV7Hs8y7aIA4RFfRIwxjrDY480a57PAfl0Nm5HEYcbz3Rsm1Qtcd/e+kW6+EFCrojBB2Clr8SMHDOfRCT6C+AvcHttsccf7b90dj7nm2884QUfAUdYOK5tiHSI9A7l1sBHjs/+PHHnWG371jO/spGfvO3p4htq+P9qNmcNSWbs6ZkM29MJi+tK+DM363h2Q2VVGXPo3T9vznlgVVM+ekiaj5fzst1U3g3apb/28jO5ZA3D5IG+9h3HjSU2ZAGWFGNTfIPVQXw4IrtGODv35hD0qhZRBsXOz9fyzOv/Yu09mrSZlzEurvO4Z/fPZFvnjqGT1ptY/afn3+Vu173yT5yu224aPyZMGmBDRN50ht3rIBh0yBtOJsP1PDJ3iqaM4/l1ORCXv+0kLN+9zaV+7dZLzwqmgPVTXz7qXVUNrSxZnsptzz3KXN+sZLrH1/Lq+sLmZaXTlZKXHdPt0+ooEcaxZug7oCNG8alemN7hwtj4C+nwDsPHt7z9oKGFhdut+m+oi+1B+y9nXieFU/oHEbpENIAQc8c5+9B71gGydk0DpuDGXuaLTe9tAes8DWUWXsmng/trZjdq6n7Yh3u2GT2McyGGTyMtQJtdiyjvsVlzzv6FBsWCoYTLmrbupTz//cdsopW0y4x/PT7N/GrhdP51cLp/N9VM1nyg1OZNDSVn772OX/cP5bs5j2cmlXPz2bUky4NVOaeyd9LJnrfRmqL7L0KeDNoHnWG9dp3Ovdjx3IYczoN7uBtQJ8X1vDap4XccPIY8gYlsfCCCwF48d//pnLDv3EjnLbgSqKjhLmjB3Pngik8d+tltCcM5uLsMp75cB8rt5TYgxVtgPoSXOPOpWXcufaHefdqm/Wy/8OOZ/7Mh3tJiI1i+JQTyGrey/KbZ5GVEkflvi2UxObS2Ori20+to6m1nee+fTzr7jqHl797Ij85fxJPXD+X9T87h79eO4fjx2YGvaZDRQU90vB4e+PPsUJyuD30umIo2WQ7cxyBtLrczH9wDT99vZe5yB4xnnieva+Dx3X2rIs22lf2QWM67z/hPCtmzTWQv4raEWdy3M/f4omySVC9F8q29/5idnrCFmfDyBMgPp2Plj/Pjg3vsrZlBKc9+DZnPriGplanl2hCGow6idrPFnPJz5+B8h1dh1t87I6t2kmOKeXawduIHnMKccn+46VMGpbKCzeewOPXz+W8L38DgF8dW8TVGVshKpbrvnY9hRmzaSYO945l3jcVn3M/uHw7p/zfBtpy5th7XbYNavaxLn4ex967nP95bRNVDd7Uyna34f7FWxmUFMdNZ9i4dWrOBFyxKYxq2cmXEjdB3hwkOcv/ekSIzj2O2XH7mDwslf95bRM1jW2wcwUG4erVqZz0TD2tMamYHcu9yQUTzqOmqY3XPz3AJTNySRg5GzCMa9/Dyzcezygp4bV9iVz+lw/YWlTL/101k4lDU4mOEuaMHsxN88czf1I2CbH9O7aPCnqksWM5DJ9lOzVkjj/8gu7xUst32MGhjjA+/qKSsroWnv94H2u2l/Z8x50rbBx4iG2IY+J58MV/bEaRh6KNkDO9c765p357i82Lbqlhccs0MPBcpT3emjeepsXVy+75O5bZOHzSYIiOpTLnZMZWvce06P0MHjeHH583ifL6Vv698UDHLmbCuaTX7+JK/OPCXfF58gkA/HrEB8RW7ewyni0inDEpmxPmzvW+jexcAaNOIi45nR8tmMG77cfQ+PkSW56WZ3tuAvsqGvnrO7sor29llXum9Zadnq13bxnO0NQEXly7nzN/t4Y/rdrJf7+4gTm/eJMPdlfwX2dPIC3ByaiJiiImdyZfHbSDcW07ifK8SQWScxxRZVv53cLJVDS08v/e2Ezz5iVslglsqY1j1JA0lrccQ/XGxVRtWGSzlnJn8conBTS1tXPtiaN80iQ3MshVQiwuEoZNZEtRLf9zwRTOmJwd/Nz9jAp6JOFpbPN4PpnjoHq/d/S3w4EnDxs6hySOAFZuLSE+JopxQ5K589VN/iGJrmhrtvHfied5xdoj0LvfBqCxqQl30abO4RYPI0+0IbCPHsFExfLQ7jwump7Ds7ddxoH4cSTseYunP+hyMvfO1BX7hS2MMTxdMYVsqSbONDNhxincNH8cE4em8NSHezBOSGdjkhXo62OWs8udw/tVXY9OaIzhnvea2ctwZhe/5L3u7ph4vk2lLNvWEapYcOww8jNOIaWpELNjmd+9fGD5NmKiorhoeg5/2O+83Xz8KKVJE9jSkMoj185m8Q9OYUJ2Kg+u2MHq7aWcMSmbR742i2tPCEj/y5lBfN2+g9uaMwPcLo6JLuDm+eP4z/rNJJRt5B2ZzYvfOYFXvncSOXO/zCBTTUr+G6yPn0tlUzvPfLiXmSMzODY33XYsSs62z8Bxmr52wRks/eGpfPOUIG9ohwkV9EjC09jm8boyx9t1T3bF4cAzjknmhL439vUTxhhWbi3hlPFZ/O6rx1FS28wv3tjS7X6FG96EtkaWtU7n7//ZzcdfVMLIk6xAO9f4+L+WE+VupSRlst++be1uVmwupk1ibCOj20XxoFmUtsTytRNHkZ2awPC5lzA3ajvrtu/pdO4PdlXw9o6yzt67J2zheMxLNhXzZNkEjKcn4vDjEBGuPXE0nxfWsmF/NQB/2xzFXnKIoZ2PY+dy/5KtXbYnLPu8mE/2VtE0+izE7bLfp2B51oFMONebSumIqohw8oKr7LLbhXG+o5/srWLxZ0XceNpY7rvkWPbHjKEyegi4XbxafwwXzxjOcSMymDwsjRe/cwJv/3g+6+46h99fcRznH5uDBL4NeX5Q03Jh6LEExce7vuXMCVyTuQOAhVfcwDHD0xER5pz1VQxCrLTzeNlETnlgFbvLG/j6ic4PiIg9zoENUGHfRGOGTGRKTlpnmw4j2uMkHDAG/nWLHa9iYsAr8pv3eOO79SXWa/BkN3j++Sp3deTs9ojK3fDGf9tejOm5vbO1aKPt/pwyFD5+1GZdxKf07hjB2Peh7aDTm7eN2ARY8Fub/wtsL6mjqKqeZwb9ndGLdvNRRguVn7VSkh9LlPNPmBIfQ6JPnNMAieVFNJk4fvhRKi1sJSZKWPGj0xg7bj5sfBHXng/4cnkFCLxRNgTf/n9PfbCXn7+xhauPH8n9E89Dti7i9YZjOTY3jZkjMmyliecR/e7v+en+7+B+eDBRjii7jSGzrB5jYJ9AUlwMcTHWB0tpK8edMJS6hHEMcrXzwLJtZA/LheTZNgc90475fenMXB5Yuo2nP9jL8IxElm8u5usjTmNUyYuMPOHLbH6rltc3FLJwVp7frWt1ufn1sm1MGprKhFMuhz1PHzR90A/P20hKtt8PwLSpx1AYP47M5n3c/E4idw6q5/7FW8hOjec7p48lKS6Gm8+cwJI3p/O1mLdY5Z7F786b1LG/iDAqM/ng5/aI9YRzg4e+wHaIik+HVT8n7qO/8kN3MSZ1OMMmzvXWSc5E8uZC4Sd8/9s3UrqsgOLaZi6Y5jMkQM4MG2Mv2eS93gFGBT0cqCuGDc/YLAFfQXe7bXfu1GF2DIysCTD5Qu/YIJ5ODr2No298wbbwr/gpfOWJnu9XXwa1hfaLPmw6fPAnG6qYclHvzh+Iq9X2ImxtsDHjnrL/I1h0ix3+IDqWt7aWck30SkYfWAzjz2ZwZiK79lTxRaNtbDMGElqiOX3iEKIcLahubOVDVyqDjzmTtV++iOqGNhb84R0eWLaNv57xXyDR5BfV8IU7nY+TTuaxLdFcf7EhKkpwu01HVsRzH+1j6uDjOGvKdfz103ncedYoryeXN5c9469ly/btJCcNZnCSTWkrrW0h313FxKGpNLe1s7+umdYWm6NtyGJp4zwW/XoVuRmJFFY38dQN85CYn0H5zo7ewSnxMSyclcsLH+8nJSGGdmPIO++/IH8oJ55xMdO3f8Rvl2/nvGOGkRzvlYM/rdrJ3opGnrxhHtFjBsFJP7CTNfSEmDhY8GvbUSiAYQt/xUcbPuPjrc2c+9DbuA08cNk0kuLsua87aTTXv3cptY3JzDrpLEYMTur58wb7Q3bqrTDjqq7riMBZP4Mv3rGrWRNgypc6/wCc+VMo287EUSN48TsjMMb4e9/Dj7Pjx2xbAplju/4BOZwYY7r9A84HtgP5wB1Bto8C3gI+A9YAed0dc/bs2UbpIduWGnNPmjF/mOlfXp5vyz95qut9HxhrzL9u6d35HjnNmHsH2WPvfqfn++1407tPW4sxv8zr/bmD8e7/2uPuWNG7/bYutvu9/7Axxphr/7jY1N2bY8yTFxvjdneqvmpriRl1+xvmn+v2d5T94Pn15ti7l5n65raOsj+u3GFG3f6G+Wh3haluaDXH3L3M3PzsJ2bRhkIz6vY3zLs7y4wxxryzo9SMuv0N8/K6/ebbT641Y+54w5z30Ntm+r3LTWOLy+/cVQ0tZvQdb5g/rNzRUXbrSxvMtHuWmVZXeydb3W632VZUax5evdNc9uf3zG0vbejyNuworjWjbn/DjLr9DfONxz7y27b2iwoz5o43zLefXGva2+092VRQbcbeudj86MVPuzzmoVJa22xue2mD+eYTHxtXu/+zWL2txFz9tw9MdWNrv50/JFTttd+ve9KM+ef1h+20wDrTha52G0MXkWjgYWABMBW4SkSmBlR7EHjKGDMduA/4VQh+axQPnsyRyl3+Yzn7DgTVFZnjO2J8PaKu2B731FttVsfS2709C7u107Fn2DTrpY07w3bW6EuOtYfaIpsZMnGB7YreGyYtgHFnwZpfUVa8jwWlj5JIMyz4TVBvav6kIUzJSePPa/JpdxvK6lpYsqmIy2bn+Xmv3zp1LMPSErh/8RaeeH8P9S0ubpo/nnOmDiU1IYZXPikAbLglMzmOi2bk8NAVxzFpWBrbiuv4yuw8EuP809cykuKYmpPG+7vKAZuWt3pbKWdMziY2uvO/qYgwaVgqN80fz8vfO4nffqXr78CEoamcMNZ24AlsRJwzejB3XTiVFVtKeHDFdlpdbm7750Yyk+O456Jjenije8+Q1Hh++5UZ/P0bc4mO8n8W8ydl8+y3TiA9Mbbfzh8S0kd4R5bsqsv/YaYnjaLzgHxjzG5jTCvwAhA4SeFUYJWzvDrIduVQ8B1MqPgz7/KBDRAd502lC0Zvc9E9jW3HfBnOux9KN9tBnXpq56Axdi5FsDHXuiJ/m3vLyntsN/bz+jBRhIgd6a6tCfPS9VwRtYbqY6+3Q5cGrS7cfMY4dpc1sGJzMS+t209bu+FrASKYGBfNbedNYmNBDf+3aidnTs5m6vA0EmKjuWh6Dks/L2ZHSR1vbS3hirkjiI+JJjk+hr9/Yw5fnZPHt08bG/T8J43LZP3eaprb2tmwv4qKhlbOmjK099cdhB+fN5mr5o1k/qTOcd7rTx7NVfNG8uc1u/j6Yx+xrbiOXy2cRnrSES6oA42nYRQ6j7I4QPQkhp4L+I7wVAAETlC4EVgI/AG4FEgVkUxjTIVvJRG5EbgRYOTILsaTVjpTtNGO9PbF23bZ0xW6aCMMPcY7Q0owMsfZCXRb6rruEejLjuXeHOHsqbZb+Or77WBMgZ00gtk5fKZ33eNRL73dZr70lvY220Hp1Ft7ll0RjKwJcML3yH7/j1RIBoMv+NlBqy84NocxWTv40+p8qhpaOXl8JuOzOzfqXjozl8fe/YItRbXcfIbXtstm5fH8x/v57jOfAHD18d7veW5GIr+5vGtP+qRxWfztP1+wfm8V/8kvJyZKOH1iaAZxmj1qELNHDQq6TUS475Jj+KK8ng93V7JwVm7IfkginpzjbDtRGAl6T7gN+JOIXAe8AxQCnXpJGGMeBR4FmDNnziG8hx9FNJTbcc1P+K5t7PJ468aZH9F3qNFgeL5olbsPHpoBm0Gye40dq8QTkljwGztu91v3wcV/7Hrfpirb43H2dd6ylGyYfoXN1e5r6uToU62g95DGVhdtLvvVana18/6uct4tOZdL3avZO+4arkns3FDnS3SU8L3Tx/GTV+xbxd1fCh52iI4Sfn/FDD7cVcHsUd7xSGaPGsTozCR2lzVw9pSh5A3qeaPe3DGDiY4S3t9VwcotJRw/dvBhCzvERkfxyNdm89zH+zq9kSgHYcrFsP/jzhNzDBA9EfRCwHdqjTynrANjzChjLvEAACAASURBVAGsh46IpACXGWOqQ2XkUY1vnDxnhlfQq/fZmXO6E2nfTJfu6u59D1rrvWOVgA1PHP9dO9ntnOv9PXA/Ox27hgcMCLXw0YOfMwRsL67jzS3FrNxaysaC6k4h+6yUeGKm/5kfnx881BLIl2fm8tBKm5t89pSuU9EmD0tj8rA0vzIRYeGsPH7/5g5vznIPSYmPYXpeOq99WkhhdRNXzTu8b7EZSXHcNP/I8DTDhrzZcMPSgbaig54I+lpggoiMwQr5lcDVvhVEJAuoNMa4gTuBIKPRK33CI5TDpltB3rncpu/1pEEU7Ahw0LNhdHessJPajj7Vv/z0n8BnL9mp1G5YHnzKtA47u7EnhBRWN/Hzf29h2WY7FOmMvHS+f+YEMhyvNkpg5shBTMtNJyqq5yllcTFRPHbdXNzGEBOkQbI7vn3qWCYOTeHUCd2EqIJw0rhMHl5tn9XZGvZQekm3gm6McYnILcByIBp4zBizWUTuw6bPLALmA78SEYMNudzcjzYfXRRttB0hEjOseBu3ndOxaCNExUB2N5kIcUk2Jt4TQd+5HMacZvfxJSHdzmD+r5tg00sw48rgdqaP8J/erJ9wtbv56zu7+b9VOwG49ZyJXDF3BNlpCSE7x5SctO4rdUFiXDTnH9u3uVxPGpfFw6t3MWloKiMze5mDrRz19CiGboxZAiwJKLvbZ/ll4OXQmnaUYIyNW48+xTttly9FG709Pz3hjKKN9m/IFNsbsjsyx0LhOvg0yDRjHlpqbZz9hJuCb59xlZ3B/s27nW7dAR7vvo86h1v6iYdW7uDh1bs4/5hh3HXRlF7FqY90Zo8aRFpCjH+PREXpIdpTdKAp2QxPf9lO3nvGnf7bmqrsJAqz7JCkpOZA8hAbbjmwwT/WfTCGz7K94v7VhVh7iI6zudvBiIqCC34Dj19oe20G48Rujt8Fu8rq2VPewJmTs7sdB+OzgmoeeXs3l8/O48GD5F6HKwmx0ay+bT5pR3oOtnJEooI+0DSU2c/3/heOu9p/8thiZ0YVT5zck/e68007H2N38XMPZ91jZxjvroNPfKr/DDKB5M6GW7f6d27yEBVtB0TqJfUtLr7+j48prG7i+DGDue+SY5k0LJXa5jbe2VFGZUMrl8/OIykuhhZXO7f9cyNZKXH87KLAvm2RQ2ZK/ECboIQpKugDTYsjjq5mO3bKFc94tx3wNHz6hDJyZjijKtJzQY+KgowQZUwkDvL2jgsBv166lQM1TXxv/jie/3gfF/zxP0zPS2dTQQ0uZxTAv769m59dNIVNhTXsKKnn8evmHvm9CBVlAFBBH2g83u6sb8D6J+3M4+POsGXBGho94i5RMKyL4UGPQBpaXDzx/h5iooRvnDSahNho3s8v55kP9/GtU8Zw+/mTufHUsfx2xXY2FdTwrVPHcvaUbFxuw72LNvPdZ9YDcPnsvAGbPEBRjnRU0Acaj4d+xk9tT9ClP4ErnwMEDqzv7IV71rMmQlw3Q4keARhjWLKpmF8s3kJRjZ0g+NmP9nHHgsncv3grY7KSufVcmx8+KDmOX146rdMx3vj+KTz1wV7e31Ue0aEWRTlUVNAHGo+HnpwF5/8anr8S/jTHu/24a/zrZ4y0DaO5sw+fjX0kv7Seexdt5t38cqbmpPGnq2fS0ubm7kWbuenZ9YjAP79zYqeBqgKJiY7ihlPGcMMAzgSjKOGACvpA01JrB8ePirYZJtcvhRqnI25UdOcRBkXgusV2nsMjlIYWF/+3Kp9/vLubhNho7rvkGK45flTHqHpLfnAqz360l8TYaOaMPkgjrKIovUIFfaBprrWzsXsYdVL3+3QxWuBA4HYbXl5fwG+Xb6eszn82octn53H7+ZMZkuqftREXE8X1J6u3rSihRgV9oGmpgfi+90ocSDYfqOFnr3/O+n3VzBqZ0TH2iACnTxrCrJGhy4ZRFKV7VNAHmkAPPUzYV9HIpX9+n9T4GH57+XQum5XXq/FSFEUJPSroA01LLST1fhCngeav7+wCA//+/ikMz0gcaHMURaFnMxYp/UkYeuiltc38c10Bl83OUzFXlCMIFfSBpqU27GLof3/3C1xuN989PfhUaoqiDAwq6ANNmHno1Y2tPPPhXr40YzijMo/8jk2KcjShgj6QuFqgvSWsPPQn399LY2s735t/ZMxyriiKFxX0gcTTSzQhfUBO/3lhDfcu2kxNU1uP6m8tquXx97/g7ClDO029pijKwKNZLgOJZxyXfvLQG1pc/GdnOau2lRATHcV9Fx/TMaVaY6uLm55dz77KRj7cXcGTN8xjaBcz/tQ2t/H7FTt4+sO9pCfGcuu5E/vFXkVRDg0V9IGkucZ+9kMM/R/vfsEDy7bR6nKTEh9DfYuLuOgo7r3YTln3m2Xb2VfZyG3nTuQva3ax8M/v8+QN8xifneJ3nH0VjVz2yPuU17dwzfEjue3cSWQkxYXcXkVRDh0V9IGkHz305ZuLGZ6ewC8XTmPu6MH8euk2/vHuF0wcmsq4Ick88f4evnHiKG45cwLzJ2Vz3eNr+coj7/Pv75/iN6Xbr5ZupaHFxb9uPpnpeRkht1NRlNChMfSBpB899APVTcwYkcFJ47KIjY7ify6YwvxJQ7j7X5/zgxc+ZeTgJG5fMBmAY3PT+ed3T6TV5ebOVzdhnJmN1u2pZOnnxXzntHEq5ooSBqigHy7ammHTy/7TwDX3j4fe7jYU1zST69PpJzpK+ONVMxmTlUxJbQu/uXw6SXHeF7QxWcncecEU/rOznBfW7scYwy8Wb2VoWjzfPk0H0lKUcEBDLoeLz1+xkzRnT4WhziQNnpBLiD300rpmXG7TqRdnWkIsz994ArtK6zl+bOfhd685fiRLNhVx/+Kt1Da1sWF/dSfhVxTlyEU99MNF+Q77WXvAW9ZPHnphVRMAuYM6d8vPSokPKuYAIsIDl03HGMOvlm5j8rBULpuVF1LbFEXpP1TQDxeVu+xnfbG3zHdyixBSWO0Ieh/GWRkxOIm7LppKTJRw14VTOyalUBTlyEffpQ8XFY6g1xV5y/qp279H0Ps6cNZV80Zy4fQc0hJiQ2mWoij9jHrohwO3Gyp32+W6Em95P01ucaC6ifTEWFLi+/57rWKuKOGHCvrhoLYQXHbGe7+QSz956Aeqm/sUblEUJbxRQT8cVOTbz5iEAA+9f4bOLaxq0nHKFeUoRAX9cOAR9Ly5h8lDbyIvSIaLoiiRjQr64aByN8QmwfDjrIfu6VzUDx56TVMbdS0uhmcEH2hLUZTIRQX9cFCRD4PHQWqOHf+8udqW94OHfqAjZTGpm5qKokQakSPoJVvgd1OgvnSgLelMRT5kjoOUoXa9rqTfJrfwdCpSD11Rjj56JOgicr6IbBeRfBG5I8j2kSKyWkQ+FZHPROSC0JvaDWVboe4AVO097Kc+KO1t1qbM8ZA6zJbVF/fb5BYHarruJaooSmTTraCLSDTwMLAAmApcJSJTA6rdBbxkjJkJXAn8OdSGdktbk/PZeNhPfVCq9oJpdzx0R9DrSvpt6NzCqibioqPISo4P6XEVRTny6YmHPg/IN8bsNsa0Ai8AlwTUMYBHmdKBAxxuPILuyfc+UvBkuGSOh1RPyKWo34bOLaxuYnhGAlHaZV9Rjjp6Iui5wH6f9QKnzJd7ga+JSAGwBPh+sAOJyI0isk5E1pWVlfXB3INwpHrovoIenwqxyVAfOg9984Ea9lY0dKxbQddwi6IcjYSqUfQq4AljTB5wAfC0iHQ6tjHmUWPMHGPMnCFDhoTo1A4ez7ztCPPQK3dB4iBIGmzXU4dCnW8Mve+C3tbu5uv/+Jibnl3fMSnFgeom7SWqKEcpPRH0QmCEz3qeU+bLN4GXAIwxHwAJQFYoDOwxHSGXpsN62m7xpCx6SBlmPXRPyOUQPPS3t5dR0dDK5gO1rN9XTavLTWldi3roinKU0hNBXwtMEJExIhKHbfRcFFBnH3AWgIhMwQp6iGMq3dARcjnSBH2XDbd48HjoIZjc4pX1BWQmx5EaH8PTH+yhuKYZY/o2bK6iKOFPt4JujHEBtwDLga3YbJbNInKfiFzsVLsV+LaIbASeB64zxneutcOA6wgU9NZGOzCXn6DnOB76ocXQqxtbeWtrKZccl8tls/NYsqmYzwpthyVNWVSUo5Meja9qjFmCbez0LbvbZ3kLcHJoTeslR6KH7hkyN3OstyxlKLTW25z5Q5jc4t8bD9Da7uay2bnEx0TxxPt7+MPKnUDfx0FXFCW8iZyeogOZtli2Aza/1rncN8PFg6dzUXn+IYVbXl5fyORhqRwzPJ3x2amcNC6TnaX1AOSkay9RRTkaiRxB78hyGQAPfe3f4NXv2IksfKnaYz8HjfGWebr/l+/oc7glv7Sejfur/eb7/PqJowA7Z2hCbGintFMUJTyIHEEfyJBLa4Mdl6UhoB24psB27ff1xD0eemN5tx56q8tNsKaIV9cXEB0lXDJzeEfZ2VOGkpOewIjBGm5RlKOVyBP0gUhbbHU69tQU+JfXFED6CP8yj4cOB/XQm9vamffLlTz70T6/8na34bVPCzltQhbZqd7QSkx0FI9eO4efX3Jsny5BUZTwJ/IEfSA8dE/v1Jr9/uU1BZCe51+WOAiinXFWDuKhl9Q2U93YxtMf7PXz0j/YVUFRTTOXzc7rtM+0vHSOzQ3tYF+KooQPkSPoA5m22OoR9EAPfX9nD13E66UfxEMvqW0BYHtJHZsP1HaUv7K+gNSEGM6eMrSrXRVFOUqJHEH3dPkfiCyXtiCC3lJnJ7II9NDBO0jXQTz00jrvdbyy3h63vsXFss+LuWj6cG34VBSlExEk6I3+nwNxbt+Qi0fcgwl6Lzz0k8ZlsmjDAdra3SzdVERTWzuXzw4cG01RFCWSBH0gB+dqPZigj+hc35PpcpDJLUprm4mLieKbp4yhoqGVNdvLeGV9AWOykpk1clCIDFcUJZKIDEF3uwc2D70tSJaLR9yDhlwcQT+oh97M0LR4Tps4hKyUOB5enc+HuytZODMXER3rXFGUzkSGoPvGzfszbdHtBnd75/K2JpAoaKzwbyCVaK94++KZueigWS4tDE1NIDY6ikuOy2XDfjtOy6WzNNyiKEpwIkvQo+P6N+Tyxg/hpa/7l7ndNoaeYXtqUuuMLFxTAGm5wcdqyXDCMMldjwlfUtdMdppNb1zoiPgJYweTNyjpkC5BUZTIJTIE3dMomTjYLvfXQI/l+XY4XF88bwRZE+2nJ9QSLAfdw5jT4drXIXd2l6cqrW3p6Dh0zPB0fnDmeG47d9KhWK8oSoQTIYLueOWJgwAD7a39c57WOjtSol+Z82MyxBHbao+g7/d64oGIwLgz7GcQGlpc1Le4GJrm7Qn63+dOYs7owYdivaIoEU6ECLojqp5p3vordbG1obOge86VOd7G0WsKbJy99kDXHno3lNbZlMWhTshFURSlJ0SGoLt8PXT6L47eUm//fPEIekKanbyipsBOYOF29VnQS2qt/b4euqIoSndEhqAfTg/d3QYun5COJ+QSm2QFvGb/wXPQe4BX0NVDVxSl50SIoHs8dEfQ+6P7v9vtzTf3Dbt4yjoEveDgOeg9oNTpJZqtHrqiKL0gMgTdk2nSnyEXj3CDv6B7PPQ4R9BrC6HaGfI2rW854yW1zSTGRpMa36MZAhVFUYBIEXRP79D+DLn4xs59lz3nik22IZb2Vihc33lii15QUtdCdlq89ghVFKVXRJag92fIpbULD71D0BO9IZZ9H/Q5fg5Ot/9UDbcoitI7IkvQ+9NDb63zWQ4Wckn2inhD2SEJemmtt5eooihKT4kMQXcFeOj9EUPvMuQS0CjqoY8NosYYSutaNGVRUZReExmC3tZsO/XEp9r1/higyy/k4rPc1gSIDbkkpEOcY0MfBb2+xUVja7umLCqK0msiRNCbrIccm+hdDzV+YRbf5QZ7bhH75xHyPncq8vQSVQ9dUZTeERmC7mqCmISBEfS2Ru95wUfQ+xZDL3U6FWVro6iiKL0kMgTd46HHJHjXQ01XMfTWRpuD7uFQPXRnLlFtFFUUpbdEkKAn2JBHTGL/xtDjUgJi6I02B93DiHmQMTL4xBY9QEMuiqL0lcjoiuhq9oY9YhP7J8ultc6+BcSn+acwtgV46Mddbf/6SEltM8lx0aRoL1FFUXpJhHjojdYzB0fQ+8lDj0u2f4Ehl9jQzSKkKYuKovSVCBH0ZhtyARtH74+QS0u9DbfEB4ZcGkIr6NqpSFGUPtIjQReR80Vku4jki8gdQbY/JCIbnL8dIlIdelMPgqdRFOxnv3noKU4M3TfLpck/5HKIlNSqh64oSt/oNlArItHAw8A5QAGwVkQWGWO2eOoYY37kU//7wMx+sLVrPGmLYD31fhH0Ouudx6VAbYFPeUCjaC9pcbXz8OpdjBuSzPyJ2XYcFxV0RVH6QE9a3uYB+caY3QAi8gJwCbCli/pXAfeExrwe0tbs46En9s/gXC31dqyYwBh6W4N/HnovWbmllD++tROA6Cih3W3ITtWQi6Iovacngp4L7PdZLwCOD1ZRREYBY4BVh25aL2hr9ImhJ0JzTejP0dpg0xEDY+iBeei9ZOXWEgYlxfK3r89h9fZSPtlbxcnjs0JgsKIoRxuhzo27EnjZGNMebKOI3AjcCDBy5MjQndXVHBBy6Y+0xfrOMXR3O7S39Dnk4mp3s3p7KWdOzmbO6MHMGT04hAYrinK00ZNG0ULAtx97nlMWjCuB57s6kDHmUWPMHGPMnCFDhvTcyoNhjOOh93ejaL03ht7WaMW8zWe2oj7wyd4qqhvbOHvK0BAaqijK0UpPBH0tMEFExohIHFa0FwVWEpHJwCDgg9Ca2A0u27OyX9MWjfFPWwQbdmn1mdyiD6zcWkJcdBSnTQzRj5uiKEc13Qq6McYF3AIsB7YCLxljNovIfSJysU/VK4EXjDGmf0ztgo4Zg/rRQ3e1gGn3diwC67F3jIXet5DLW1tLOWFcpvYKVRQlJPRISYwxS4AlAWV3B6zfGzqzeoEno6U/0xY9MfP4VO94560Ndv5Q6FPIZVdZPbvLG7ju5NGhsVFRlKOe8HcNPeLtCXvEJFpvur0NomNDcw6PoPt66C11No4OffLQ39paAsBZGj9XFCVEhH/X/0BB7xgTPYTzinryzgNj6B0hl97H0FduKWVqThq5GX3PYVcURfEl/AW9I+TiEXTPmOghTF3sGDo32Yo6WK+9tfdZLsYYNuyvZt3eSs6ekh06GxVFOeqJgJBLQKaJp3E0lB66Z7jc+FQfQffpXNRNyKW5rZ33d5Xz1tZSVm0rpaimmYTYKL40Y3jobFQU5agnAgTd8cQ7YuiOhx7K7v8tPjF0T8ilpQ6iop3y4B56UU0Td/9rM//ZWUZzm5ukuGhOnZDFj86ZyBmTshmiXfwVRQkh4S/onpzzjiyXfphX1PHG397TRGW7cCnYkEuU0+jaxfC5f3wrn7d3lHHl3BGcNWUox48ZTEJsdOjsUhRF8SH8Bb3LRtFQCrr10J9aX857Be1cGosV+eiuBb25rZ03PjvARdNyuO+SY0Nni6IoSheEf6NosLRFCG1vUUfQCxuiaW6Htqh4G3JpawKJgpjOoZM3t5RQ1+zistl9myxaURSlt0SeoPeHh95iwyvFjbYTbE17Ai2Ndd6x0EU67fLK+gKGpydw4tjM0NmhKIpyEMJf0Dti6IGCHtq0RROXTHVjGxdOy6HeJLC7sLjLsdBLa5t5Z0cZl87KJSqqs9griqL0B+Ev6G3NgHjDHrH9E3Jxx9rslhPHZRKdkEJxeQVtzfVBM1xe31CI28DCWRpuURTl8BEBgt5oRdwT9ojpn0bRthgr3FkpcQwaNJj49kYKSio65aAbY3jlk0KOG5HBuCEpobNBURSlG8Jf0H0ntwCfnqKhjaG3RtkfisHJ8aSkpjM0wUVpZRUmwEPffKCW7SV12hiqKMphJ/wFva3JP22wnzz0ZrHHzUyJg7gUhsS3EeVqpMHtPwDYa58WEhstfGl6TujOryiK0gMiRNB9PPSoKIiOD3EMvYFGR9CzkuMhLoVkmkmihdJmb0cht9uwdFMRp00YQkZSXOjOryiK0gPCX9BdzZ0zTWITQ5vl0lJHg4knJkpIS4yB+BSi2xoZFNvG/npvFsuGgmoO1DRzwTT1zhVFOfyEv6C3NXrDLB5iE0M8OFcDte4EMlPiEBE7pktrHWnRbRQ1RlFcY388Fn9WRGy0cPZUHeNcUZTDTwQIerN/yAWceUVDmYdeT017HJnJTmpkXAoYN0nttTQRz1vbSvzCLemJIZpYQ1EUpRdEgKA3dh5LJZTzira7wNVMZVucbRAFO4wuENXeQmxCCm9tLdVwi6IoA074D84VmLYIoZ1X1BnHpaItlsxkR9DjvLnnw7MH825+OcPSEzTcoijKgBIBHnpT/3roztC5pS2xZKb4hFwcxuRk0+py88LH+zTcoijKgBIhgh4sht47QTfGYIzpvMHx0Ktc8d6Qi4+HPmJYFqkJMbgNGm5RFGVACX9BdzUHyXJJ6HXa4rkPvcMjb+/uvMER9AYSbA46dMTQAWLikjljUjZx0VEablEUZUAJb0E3xjuWiy+xSb1KW6xpamNnaT1rtpd23uhMP9dgEoJ66MQl8dMLp/Dct4/XcIuiKANKeDeKtreBcR9y2uK+Civ+mwpraHcbon2HvPXx0IPF0IlNYmhaAkPTAmxQFEU5zIS3h+7xwg+xUXRPhW34bGxtZ1dZvf9Gp1G0gQRvlotPyMXPW1cURRlAwlvQPV74IaYt7nUEHWDj/mr/jS11wEFCLkEmuFAURRkIwlvQA6ef8xCTCO422ymoB+ytaCQrJZ7U+Bg2FgQIuuOhu2NTSIpzIlQx8RDlmSBaPXRFUY4MwjuG3pWg+85aFJ1Kd+ytaGRsVjLRUcJnBTX+G50YelJywGQVccnQXB10xiJFUZSBILw99MD5RD10Ma+oq93Nff/ewu6AOPneygZGZiYxY0QGW4tqaXG1eze2NtAkiQxODTiHJ44eGL9XFEUZIMJb0Lvz0ANSF7cW1fHYe1/w8icFHWVNre2U1LYwOjOJGXnptLUbthbVeXdqqaOJBLKSA8Y398TRVdAVRTlCCHNBdzzwTjF0p5E0IHXREx/3Davsq7SiPzIzmekjMpztPnH01gbqfRtEPcSlQFQMxOhEFoqiHBn0SNBF5HwR2S4i+SJyRxd1vioiW0Rks4g8F1ozu8DVlYfueM0BmS6eDJaNBdW43babvydlcXRmEsPTE8hKiWODT6aLaa2j1vjkoHuIT9EGUUVRjii6bRQVkWjgYeAcoABYKyKLjDFbfOpMAO4ETjbGVIlIdn8ZTPEmKFhnlws/sZ/Buv4DbH4VDnzaUZy7eyfXxLTgbjNUvPMFQ1LiSdxZxlXRxYzfV4QUR/P99D1U7m6FdXY/d/kuGky8NwfdQ1yKNogqinJE0ZMsl3lAvjFmN4CIvABcAmzxqfNt4GFjTBWAMSZIH/oQsWsVvHm3dz0mEZKz/Ouk5YFEwXt/8Cv+L/Be8Rr7cRpwWiywwq5/w1P5DfsRDex1n9455JI1AZoDMmIURVEGkJ4Iei6w32e9ADg+oM5EABF5D6uB9xpjlgUeSERuBG4EGDlyZF/shTnfhGlf9a7Hp/j33AQYMhF+8oVfyGX9viq++8x6HvzKdO56/XMumTGcW8+dxC3Pf0pDi4vHr5sLwHu7yvnRixt5+JqZzB01mA0F1dzxVD5PJAeEXM66p2/2K4qi9BOhahSNASYA84GrgL+JSEZgJWPMo8aYOcaYOUOGDOnbmeJTIC3H+xco5h4SM/zqfVKZQCmDmDppEkNzR/NeaSyk5bCxJoHUISM66k2ZOIlSBvFJZQKk5VDszsBNVGcPXcT+KYqiHCH0RNALgRE+63lOmS8FwCJjTJsx5gtgB1bgjxg2FlSTm5FIVko8M/Iy2HyglqbWdgqrmhid6Y2FD06OY+yQZF5bX0iry01FQysAWYGNooqiKEcYPRH0tcAEERkjInHAlcCigDqvY71zRCQLG4IJMrj4wPFZQQ0zRqQDMH1EBi0ut53c2diURV/+Z8EUtpfU8adVO6mot4I+KEnTExVFObLpVtCNMS7gFmA5sBV4yRizWUTuE5GLnWrLgQoR2QKsBn5sjKnoL6N7S2VDK/sqG5meZ6NAM/KssC/acADAz0MHOHvqUBbOzOXhNbt4d2c56YmxxMWEd8q+oiiRT4/GcjHGLAGWBJTd7bNsgP92/o44PB2FZjiCPnJwEhlJsazZXgbAqMzO+eT3fOkY3s0v5+M9lYzN0nxzRVGOfI4Kt/OzghpEYJrjmYsI0/MyaG13kxQXTVZggyeQnhTLLy+dBtC5QVRRFOUI5KgQ9I37qxk3JIWUeO8LiSfsMiozGekiW+XsqUP54VkTuHRm3mGxU1EU5VAI7+Fze4Axho0FNZw20b/zkSeePmrwwXt7/uicif1mm6IoSiiJeA+9oKqJ8vqWjvi5B0/Gy6gs7b6vKEpkEDGCvnJLCcfdt4Ivyhv8yt/cUgLAqRP8PfTs1AR+95UZXHvCqMNmo6IoSn8SEYJe1dDKHa9uorqxjZfW7ffbtmRTEVNy0hg7JKXTfpfNziNvkHroiqJEBhEh6Pcs2kx1YyuTh6Xy2vpC2p2hcYtqmli3t4oLpw0bYAsVRVH6n7AX9GWfF7No4wFuOXM83z9zAsW1zby/qxyApZuKAbhgWs5AmqgoinJYCOssl6qGVu56/XOm5qRx8xnjaXcb0hJieOWTAk6dMOSg4RZFUZRII6w99BVbiimvb+GXC6cRGx1FQmw0F80YzrLNxeSX1mm4RVGUo4qwFvRyZ+CsycO8Q+heNiuP5jY3P3pxI6DhFkVRjh7CWtCrGlpJjI0mITa6o2zWyAzGZCWzqbBGwy2KohxVYmeugAAAB5xJREFUhLegN7YxOGCuTxHhslm5ABpuURTlqCK8G0UbW8lIiu1UfsXckWw+UMtX5owIspeiKEpkEvaCHmziiSGp8fzla7MHwCJFUZSBI6xDLtWNbQxK1qFtFUVRIMwFvbKhlUFBQi6KoihHI2Er6K52N7XNbTrXp6IoikPYCnpNUxvGoB66oiiKQ9gKelVjG4DG0BVFURzCWNBtL1ENuSiKoljCV9AbVNAVRVF8CVtBr+4IuWgMXVEUBcJY0Cs15KIoiuJH2Ap6VWMrcdFRJMVFd19ZURTlKCB8Bb2hlUHJsYjIQJuiKIpyRBC+gt6onYoURVF8CVtBr+5iYC5FUZSjlbAV9Eon5KIoiqJYwlbQqxvbyFAPXVEUpYOwFHS321DV2MpgFXRFUZQOwlLQ65pduA1BZytSFEU5WumRoIvI+SKyXUTyReSOINuvE5EyEdng/H0r9KZ68YzjEjifqKIoytFMt1PQiUg08DBwDlAArBWRRcaYLQFVXzTG3NIPNnZCe4kqiqJ0pice+jwg3xiz2xjTCrwAXNK/Zh2cakfQNeSiKIripSeCngvs91kvcMoCuUxEPhORl0VkRLADiciNIrJORNaVlZX1wVxLZYMdmEtDLoqiKF5C1Sj6b2C0MWY68CbwZLBKxphHjTFzjDFzhgwZ0ueTeT10FXRFURQPPRH0QsDX485zyjowxlQYY1qc1b8Ds0NjXnCqGluJjhLSErptAlAURTlq6ImgrwUmiMgYEYkDrgQW+VYQkRyf1YuBraEzsTOVDW0MStKBuRRFUXzp1sU1xrhE5BZgORANPGaM2Swi9wHrjDGLgB+IyMWAC6gErutHm6lubNVwi6IoSgA9ilkYY5YASwLK7vZZvhO4M7SmdU1lg/YSVRRFCSQse4racVw0ZVFRFMWXsBT0qsZWTVlUFEUJIOwE3Rg7MJfG0BVFUfwJO0FvaG2nrd0wSEMuiqIofoSdoFc1OOO4aMhFURTFj/ATdB2YS1EUJShhKOiecVw05KIoiuJL+Al6g47joiiKEozwE3QNuSiKogQl7AQ9NyORc6cOJT1RQy6Koii+hN1wheceM4xzjxk20GYoiqIccYSdh64oiqIERwVdURQlQlBBVxRFiRBU0BVFUSIEFXRFUZQIQQVdURQlQlBBVxRFiRBU0BVFUSIEMcYMzIlFyoC9fdw9CygPoTnhwtF43UfjNcPRed1H4zVD7697lDFmSLANAyboh4KIrDPGzBloOw43R+N1H43XDEfndR+N1wyhvW4NuSiKokQIKuiKoigRQrgK+qMDbcAAcTRe99F4zXB0XvfReM0QwusOyxi6oiiK0plw9dAVRVGUAFTQFUVRIoSwE3QROV9EtotIvojcMdD29AciMkJEVovIFhHZLCI/dMoHi8ibIrLT+Rw00LaGGhGJFpFPReQNZ32MiHzkPO8XRSTi5h4UkQwReVlEtonIVhE58Sh51j9yvt+fi8jzIpIQac9bRB4TkVIR+dynLOizFcsfnWv/TERm9fZ8YSXoIhINPAwsAKYCV4nI1IG1ql9wAbcaY6YCJwA3O9d5B/CWMWYC8JazHmn8ENjqs/4A8JAxZjxQBXxzQKzqX/4ALDPGTAZmYK8/op+1iOQCPwDmGGOOBaKBK4m85/0EcH5AWVfPdgEwwfm7EfhLb08WVoIOzAPyjTG7jTGtwAvAJQNsU8gxxhQZY9Y7y3XYf/Bc7LU+6VR7EvjywFjYP4hIHnAh8HdnXYAzgZedKpF4zenAacA/AIwxrcaYaiL8WTvEAIkiEgMkAUVE2PM2xrwDVAYUd/VsLwGeMpYPgQwRyenN+cJN0HOB/T7rBU5ZxCIio4GZwEfAUGNMkbOpGBg6QGb1F/8L/ARwO+uZQLUxxuWsR+LzHgOUAY87oaa/i0gyEf6sjTGFwIPAPqyQ1wCfEPnPG7p+toesb+Em6EcVIpICvAL8lzGm1nebsfmmEZNzKiIXAaXGmE8G2pbDTAwwC/iLMWYm0EBAeCXSnjWAEze+BPuDNhxIpnNoIuIJ9bMNN0EvBEb4rOc5ZRGH/P/27Z41qiAKA/AzhQlYGWsLEcQ2ZSAWolYprOwEU/grxMo/4D+wCsEiIehi6UftRyEqKmpQYgo/KusUx2ImkGZB0etlJ+eBy969u7AzvJcDc+5sKUfUYr4eEVvt8rf9JVh7/T7W+AawjEullM9qK+282ls+1pbk9Jn3LnYj4kl7v6kW+J6zhov4FBE/ImIPW+o90HveTM/2r+vbrBX0ZzjdnoTPqQ9RJiOP6Z9rvePbeBsRtw58NMFqO1/Fvf89tqFExPWIOBERJ9VcH0XEFTzG5fa1ruYMEfEVX0opZ9qlC3ij46ybHSyVUo62+31/3l3n3UzLdoKrbbfLEn4eaM38noiYqQMreI9t3Bh7PAPN8ay6DHuJF+1YUXvKD/EBD3B87LEONP9zuN/OT+EpPmID82OPb4D5LuJ5y/suFg5D1riJd3iNNcz3ljfuqM8I9tTV2LVp2aKou/i28UrdAfRHv5d//U8ppU7MWsslpZTSFFnQU0qpE1nQU0qpE1nQU0qpE1nQU0qpE1nQU0qpE1nQU0qpE78ArjiSFU8aA4cAAAAASUVORK5CYII=\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Plotting Confusion Matrix**"
      ],
      "metadata": {
        "id": "NsYpWyOnzQZc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "cm = confusion_matrix(y_test,y_pred)\n",
        "print(cm)\n",
        "accuracy_score(y_test,y_pred)"
      ],
      "metadata": {
        "id": "SJ30uAi3zVAZ",
        "outputId": "9543051a-248f-4f1f-efcf-0d4517a6d56e",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[27  4]\n",
            " [ 1 46]]\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9358974358974359"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def normalized_confusion_matrix(y_test, conf_mat, model):\n",
        "    _ , counts = np.unique(y_test,return_counts=True)\n",
        "    conf_mat = conf_mat/counts\n",
        "    plt.figure(figsize=(6,5))\n",
        "    ax=sns.heatmap(conf_mat,fmt='.2f',annot=True,annot_kws={'size':20},lw=2, cbar=True, cbar_kws={'label':'% Class accuracy'})\n",
        "    plt.title(f'Confusion Matrix ({model})',size=22)\n",
        "    plt.xticks(size=20)\n",
        "    plt.yticks(size=20)\n",
        "    ax.figure.axes[-1].yaxis.label.set_size(20) ##colorbar label\n",
        "    cax = plt.gcf().axes[-1]  ##colorbar ticks\n",
        "    cax.tick_params(labelsize=20) ## colorbar ticks\n",
        "    plt.savefig(f'confusion-matrix-{model}.png',dpi=300)"
      ],
      "metadata": {
        "id": "UnW9W0hjrJLu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "conf_mat = confusion_matrix(y_test,y_pred)\n",
        "normalized_confusion_matrix(y_test,conf_mat, 'Diabetes Pred Model')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 351
        },
        "id": "r9zCB6K2rqRE",
        "outputId": "0fdccb1d-8462-452b-efb9-1393b35d29a3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x360 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "j04GGyQEoG1F",
        "outputId": "5a209a74-e036-4455-da24-f26a2ed82eda"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " dense (Dense)               (None, 128)               2176      \n",
            "                                                                 \n",
            " dropout (Dropout)           (None, 128)               0         \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 32)                4128      \n",
            "                                                                 \n",
            " dropout_1 (Dropout)         (None, 32)                0         \n",
            "                                                                 \n",
            " dense_2 (Dense)             (None, 1)                 33        \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 6,337\n",
            "Trainable params: 6,337\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Save the Model**"
      ],
      "metadata": {
        "id": "Ds7OU4nizc9C"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "save as model.pb"
      ],
      "metadata": {
        "id": "DHKq5AEbrEYV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "tf.saved_model.save(\n",
        "    model,\n",
        "    export_dir = \"/tmp/myModel\",\n",
        ")"
      ],
      "metadata": {
        "id": "azt5PZJTsIJ4",
        "outputId": "f2be89cf-ed27-44b0-9672-f244dee5dcf2",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "INFO:tensorflow:Assets written to: /tmp/myModel/assets\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Save as H5"
      ],
      "metadata": {
        "id": "hBm8PpFWrBK9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('model.h5')"
      ],
      "metadata": {
        "id": "ET3jrCNZIhtp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "save as model.pkl"
      ],
      "metadata": {
        "id": "cVTQt2Y6rJ2n"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "pickle.dump(model,open('model.pkl','wb'))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8RchaCU0l2D0",
        "outputId": "836f48ef-7ed3-4616-d554-21fb57704e9d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "INFO:tensorflow:Assets written to: ram://5172d3cb-201c-4b99-b2b8-55c6b8cb91fe/assets\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model = pickle.load(open('model.pkl','rb'))"
      ],
      "metadata": {
        "id": "SmIQYhCkmqAq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "48QKMhDwnQJF"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}