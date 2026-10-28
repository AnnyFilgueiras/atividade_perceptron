from data import data_train, data_test
from perceptron import train, predict, evaluate


def format_label(class_value):
    return "ALERTA (1)" if class_value == 1 else "NORMAL (0)"


def interactive_classification(weightTemperature, weightHumidity, bias):
    print("--- Classificação de novos valores (digite 'sair' para encerrar) ---")

    while True:
        try:
            temp_input = input("Digite a temperatura (°C): ").strip()
            if temp_input.lower() in ("sair", "exit", "q"):
                print("Encerrando o programa.")
                break

            hum_input = input("Digite a umidade relativa (%): ").strip()
            if hum_input.lower() in ("sair", "exit", "q"):
                print("Encerrando o programa.")
                break

            temperature = float(temp_input.replace(",", "."))
            humidity = float(hum_input.replace(",", "."))

            result = predict(temperature, humidity, weightTemperature, weightHumidity, bias)
            print(f"Classificação prevista: {format_label(result)}\n")

        except ValueError:
            print("Entrada inválida! Digite valores numéricos ou 'sair' para encerrar.\n")
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o programa.")
            break


def main():
    print("--- Iniciando treinamento ---")
    weightTemperature, weightHumidity, bias, epochs, lastEpochErrors = train(data_train)
    print("--- Treinamento concluído ---")
    print(f"Épocas executadas: {epochs}")
    print(f"Erros na última época: {lastEpochErrors}")
    print(f"Peso final da temperatura: {weightTemperature:.4f}")
    print(f"Peso final da umidade: {weightHumidity:.4f}")
    print(f"Viés final: {bias:.4f}\n")

    print("--- Conjunto de testes (Previsto vs. Esperado) ---")
    results, predictCorrects, accuracy = evaluate(
        data_test, weightTemperature, weightHumidity, bias
    )

    for r in results:
        status = "OK" if r["correct"] else "ERRO"
        print(
            f"ID {r['id']:2d} | Temp: {r['temperature']:2d}°C | Umid: {r['humidity']:2d}% | "
            f"Esperado: {format_label(r['expected']):10s} | "
            f"Previsto: {format_label(r['predicted']):10s} [{status}]"
        )

    print(f"\nAcertos: {predictCorrects}/{len(data_test)}")
    print(f"Acurácia no conjunto de teste: {accuracy:.2f}%\n")

    interactive_classification(weightTemperature, weightHumidity, bias)


if __name__ == "__main__":
    main()
