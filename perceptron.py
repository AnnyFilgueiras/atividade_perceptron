class Perceptron:
    TEMP_SCALE = 40.0
    HUMIDITY_SCALE = 100.0

    def __init__(self, rate=0.1, max_epoch=1000):
        self.rate = rate
        self.max_epoch = max_epoch
        self._reset_state()

    def _reset_state(self):
        self.weight_temperature = 0.0
        self.weight_humidity = 0.0
        self.bias = 0.0
        self.epochs_executed = 0
        self.last_epoch_errors = 0

    def normalize(self, temperature, humidity):
        temperature_normalized = temperature / self.TEMP_SCALE
        humidity_normalized = humidity / self.HUMIDITY_SCALE
        return temperature_normalized, humidity_normalized

    def predict(self, temperature, humidity):
        temperature_normalized, humidity_normalized = self.normalize(temperature, humidity)
        weighted_sum = (
            self.weight_temperature * temperature_normalized
            + self.weight_humidity * humidity_normalized
            + self.bias
        )
        return 1 if weighted_sum >= 0 else 0

    def update_parameters(self, error, temperature, humidity):
        temperature_normalized, humidity_normalized = self.normalize(temperature, humidity)
        self.weight_temperature += self.rate * error * temperature_normalized
        self.weight_humidity += self.rate * error * humidity_normalized
        self.bias += self.rate * error

    def train(self, data_train):
        self._reset_state()

        for current_epoch in range(self.max_epoch):
            errors_in_epoch = 0

            for sample in data_train:
                output_predict = self.predict(sample["temperature"], sample["humidity"])
                error = sample["class"] - output_predict

                if error != 0:
                    self.update_parameters(error, sample["temperature"], sample["humidity"])
                    errors_in_epoch += 1

            self.epochs_executed = current_epoch + 1
            self.last_epoch_errors = errors_in_epoch

            if errors_in_epoch == 0:
                break

    def evaluate(self, data_test):
        results = []
        predict_corrects = 0

        for sample in data_test:
            predicted = self.predict(sample["temperature"], sample["humidity"])
            expected = sample["class"]
            is_correct = predicted == expected
            if is_correct:
                predict_corrects += 1

            results.append({
                "id": sample["id"],
                "temperature": sample["temperature"],
                "humidity": sample["humidity"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            })

        accuracy = (predict_corrects / len(data_test)) * 100 if data_test else 0.0
        return results, predict_corrects, accuracy


class MinMaxPerceptron(Perceptron):
    def __init__(self, rate=0.1, max_epoch=1000):
        super().__init__(rate=rate, max_epoch=max_epoch)
        self.temp_min = 0.0
        self.temp_max = self.TEMP_SCALE
        self.hum_min = 0.0
        self.hum_max = self.HUMIDITY_SCALE

    def _fit_scaler(self, data_train):
        self.temp_min = min(sample["temperature"] for sample in data_train)
        self.temp_max = max(sample["temperature"] for sample in data_train)
        self.hum_min = min(sample["humidity"] for sample in data_train)
        self.hum_max = max(sample["humidity"] for sample in data_train)

    def normalize(self, temperature, humidity):
        temperature_normalized = (temperature - self.temp_min) / (self.temp_max - self.temp_min)
        humidity_normalized = (humidity - self.hum_min) / (self.hum_max - self.hum_min)
        return temperature_normalized, humidity_normalized

    def train(self, data_train):
        self._fit_scaler(data_train)
        super().train(data_train)



