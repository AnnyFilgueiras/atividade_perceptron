def normalize(temperature, humidity):
    temperatureNormalized = temperature / 40
    humidityNormalized    = humidity / 100
    return temperatureNormalized, humidityNormalized


def predict(temperature, humidity, weightTemperature, weightHumidity, bias):
    temperatureNormalized, humidityNormalized = normalize(temperature, humidity)
    weightedSum = (
        weightTemperature * temperatureNormalized
        + weightHumidity * humidityNormalized
        + bias
    )

    if weightedSum >= 0:
        return 1
    else:
        return 0


def update_parameters(weightTemperature, weightHumidity, bias, error, temperature, humidity, rate=0.1):
    temperatureNormalized, humidityNormalized = normalize(temperature, humidity)
    weightTemperature += rate * error * temperatureNormalized
    weightHumidity    += rate * error * humidityNormalized
    bias              += rate * error
    return weightTemperature, weightHumidity, bias


def train(data_train, rate=0.1, maxEpoch=1000):
    weightTemperature = 0.0
    weightHumidity    = 0.0
    bias              = 0.0
    errorsInEpoch     = 0

    for currentEpoch in range(maxEpoch):
        errorsInEpoch = 0

        for d in data_train:
            outputPredict = predict(
                d["temperature"], d["humidity"],
                weightTemperature, weightHumidity, bias
            )
            error = d["class"] - outputPredict

            if error != 0:
                weightTemperature, weightHumidity, bias = update_parameters(
                    weightTemperature, weightHumidity, bias,
                    error, d["temperature"], d["humidity"], rate
                )
                errorsInEpoch += 1

        if errorsInEpoch == 0:
            break

    return weightTemperature, weightHumidity, bias, currentEpoch + 1, errorsInEpoch


def evaluate(data_test, weightTemperature, weightHumidity, bias):
    results = []
    predictCorrects = 0

    for d in data_test:
        predicted = predict(
            d["temperature"], d["humidity"],
            weightTemperature, weightHumidity, bias
        )
        expected = d["class"]
        isCorrect = (predicted == expected)
        if isCorrect:
            predictCorrects += 1

        results.append({
            "id": d["id"],
            "temperature": d["temperature"],
            "humidity": d["humidity"],
            "expected": expected,
            "predicted": predicted,
            "correct": isCorrect
        })

    accuracy = (predictCorrects / len(data_test)) * 100 if data_test else 0.0
    return results, predictCorrects, accuracy
