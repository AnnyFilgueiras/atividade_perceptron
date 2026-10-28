from data import data_train, data_test
from perceptron import Perceptron

EXIT_COMMANDS = ("sair", "exit", "q")


def format_label(class_value):
    return "ALERTA (1)" if class_value == 1 else "NORMAL (0)"


def display_training_summary(model):
    print("--- Treinamento concluído ---")
    print(f"Épocas executadas: {model.epochs_executed}")
    print(f"Erros na última época: {model.last_epoch_errors}")
    print(f"Peso final da temperatura: {model.weight_temperature:.4f}")
    print(f"Peso final da umidade: {model.weight_humidity:.4f}")
    print(f"Viés final: {model.bias:.4f}\n")


def display_test_results(results, predict_corrects, accuracy):
    print("--- Conjunto de testes (Previsto vs. Esperado) ---")
    for result in results:
        status = "OK" if result["correct"] else "ERRO"
        print(
            f"ID {result['id']:2d} | Temp: {result['temperature']:2d}°C | Umid: {result['humidity']:2d}% | "
            f"Esperado: {format_label(result['expected']):10s} | "
            f"Previsto: {format_label(result['predicted']):10s} [{status}]"
        )

    print(f"\nAcertos: {predict_corrects}/{len(results)}")
    print(f"Acurácia no conjunto de teste: {accuracy:.2f}%\n")


def interactive_classification(model):
    print("--- Classificação de novos valores (digite 'sair' para encerrar) ---")

    while True:
        try:
            temp_input = input("Digite a temperatura (°C): ").strip()
            if temp_input.lower() in EXIT_COMMANDS:
                print("Encerrando o programa.")
                break

            hum_input = input("Digite a umidade relativa (%): ").strip()
            if hum_input.lower() in EXIT_COMMANDS:
                print("Encerrando o programa.")
                break

            temperature = float(temp_input.replace(",", "."))
            humidity = float(hum_input.replace(",", "."))

            prediction = model.predict(temperature, humidity)
            print(f"Classificação prevista: {format_label(prediction)}\n")

        except ValueError:
            print("Entrada inválida! Digite valores numéricos ou 'sair' para encerrar.\n")
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o programa.")
            break


def main():
    model = Perceptron(rate=0.1, max_epoch=1000)

    print("--- Iniciando treinamento ---")
    model.train(data_train)
    display_training_summary(model)

    results, predict_corrects, accuracy = model.evaluate(data_test)
    display_test_results(results, predict_corrects, accuracy)

    interactive_classification(model)


if __name__ == "__main__":
    main()


