class MedAnalytics {
    constructor() {
        this.currentDataset = 'medimp';
        this.baseUrl = 'http://127.0.0.1:5000';
        this.isLoading = false;

        this.elements = {
            navButtons: document.querySelectorAll('.nav-btn'),
            startButton: document.getElementById('startInference'),
            showResultsButton: document.getElementById('showResults'),
            accuracyDisplay: document.getElementById('accuracy').querySelector('.accuracy-value'),
            resultsTable: document.getElementById('resultsTable'),
            resultsCard: document.getElementById('resultsCard'),
            spinner: document.querySelector('.spinner'),
            dropZone: document.getElementById('dropZone'),
            fileInput: document.getElementById('fileInput'),
            uploadStatus: document.getElementById('uploadStatus')
        };

        this.initializeEventListeners();
        this.clearInitialState();
    }

    initializeEventListeners() {
        // Навигация
        this.elements.navButtons.forEach(button => {
            button.addEventListener('click', () => this.handleDatasetChange(button));
        });

        // Управление результатами
        this.elements.showResultsButton.addEventListener('click', async () => {
            await this.loadResults();
            this.toggleResults();
        });

        this.elements.startButton.addEventListener('click', () => this.startInference());

        // Drag and Drop загрузка файлов
        this.elements.dropZone.addEventListener('click', () => this.elements.fileInput.click());
        this.elements.fileInput.addEventListener('change', (event) => this.handleFileUpload(event.target.files));

        ['dragenter', 'dragover'].forEach(eventType => {
            this.elements.dropZone.addEventListener(eventType, (event) => {
                event.preventDefault();
                event.stopPropagation();
                this.elements.dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventType => {
            this.elements.dropZone.addEventListener(eventType, (event) => {
                event.preventDefault();
                event.stopPropagation();
                this.elements.dropZone.classList.remove('dragover');
            });
        });

        this.elements.dropZone.addEventListener('drop', (event) => {
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files);
            }
        });
    }

    clearInitialState() {
        this.hideResults();
        this.elements.resultsTable.innerHTML = '';
        this.elements.accuracyDisplay.textContent = '--';
    }

    handleDatasetChange(button) {
        console.log(`Смена набора данных на: ${button.dataset.dataset}`);

        this.elements.navButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');

        this.currentDataset = button.dataset.dataset;

        this.clearInitialState();
    }

    async handleFileUpload(files) {
        if (!files || files.length === 0) return;
        const formData = new FormData();
        Array.from(files).forEach(file => formData.append('file', file));
        try {
            console.log(`Загрузка файлов для набора данных: ${this.currentDataset}`);
            const response = await fetch(`${this.baseUrl}/upload/${this.currentDataset}`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.message || 'Ошибка при загрузке файлов');
            console.log('Файлы успешно загружены:', data);

            // Используем showNotification вместо showUploadStatus
            this.showNotification('Файл успешно загружен!', 'success');
        } catch (error) {
            console.error('Ошибка загрузки файлов:', error.message);
            this.showNotification(error.message, 'error');
        }
    }

    toggleResults() {
        if (!this.currentResults) {
            this.showNotification('Нет доступных результатов', 'error');
            return;
        }

        const isHidden = this.elements.resultsCard.classList.contains('hidden');

        if (isHidden) {
            this.showResults();
        } else {
            this.hideResults();
        }
    }

    showResults() {
        console.log('Показ результатов');

        this.elements.resultsCard.classList.remove('hidden');
        this.elements.resultsCard.classList.add('animate__fadeInUp');

        this.elements.showResultsButton.querySelector('.btn-text').textContent = 'Скрыть результаты';

        this.renderTable();
    }

    hideResults() {
        console.log('Скрытие результатов');

        this.elements.resultsCard.classList.add('hidden');
        this.elements.resultsCard.classList.remove('animate__fadeInUp');

        this.elements.showResultsButton.querySelector('.btn-text').textContent = 'Показать результаты';
    }

    async startInference() {
        if (this.isLoading) return;

        try {
            console.log(`Запуск инференса для набора данных: ${this.currentDataset}`);

            this.setLoading(true);

            const response = await fetch(`${this.baseUrl}/inference/${this.currentDataset}`);
            const data = await response.json();

            if (!response.ok) throw new Error(data.message || 'Ошибка при выполнении инференса');

            console.log('Инференс успешно выполнен:', data);

            await this.loadResults();

            this.showNotification('Инференс успешно выполнен', 'success');


        } catch (error) {
            console.error('Ошибка инференса:', error.message);

            this.showNotification(error.message, 'error');

        } finally {
            this.setLoading(false);
        }
    }

    setLoading(isLoading) {
        console.log(isLoading ? 'Начало загрузки...' : 'Загрузка завершена.');

        this.isLoading = isLoading;

        if (isLoading) {
            this.elements.spinner.style.display = 'block';
            this.elements.startButton.disabled = true;
            this.elements.startButton.querySelector('.btn-text').textContent = 'Загрузка...';
        } else {
            this.elements.spinner.style.display = 'none';
            this.elements.startButton.disabled = false;
            this.elements.startButton.querySelector('.btn-text').textContent = 'Запустить анализ';
        }
    }

    async loadResults() {
        try {
            console.log(`Загрузка результатов для набора данных: ${this.currentDataset}`);

            const response = await fetch(`${this.baseUrl}/results/${this.currentDataset}`);
            const data = await response.json();

            if (!response.ok) throw new Error(data.message || 'Ошибка при загрузке результатов');

            console.log('Результаты успешно загружены:', data);

            const { accuracy, results } = data.data;

            if (accuracy !== undefined) {
                this.elements.accuracyDisplay.textContent = accuracy;
            }

            // Преобразуем данные для таблицы
            const parsedResults = results.map((item, index) => ({
                Номер: index + 1,
                "Название файла": item.file_name,
                Предсказание: item.prediction,
                Уверенность: (item.confidence * 100).toFixed(2),
                "Истинное значение": item.true_value,
                Вывод: item.is_correct ? "Верно" : "Неверно"
            }));

            // Сохраняем преобразованные результаты
            this.currentResults = parsedResults;

        } catch (error) {
            console.error('Ошибка загрузки результатов:', error.message);

            this.showNotification(error.message, 'error');

            this.currentResults = null;

            this.clearInitialState();

        }
    }

    renderTable() {
        if (!this.currentResults) return;

        const tableRows = this.currentResults.map(result => `
            <tr class="animate__animated animate__fadeIn">
                <td>${result.Номер}</td>
                <td>${result['Название файла']}</td>
                <td>${result.Предсказание}</td>
                <td>${result.Уверенность}%</td>
                <td>${result['Истинное значение']}</td>
                <td>
                    <span class="status-badge ${result.Вывод === 'Верно' ? 'success' : 'error'}">
                        ${result.Вывод}
                    </span>
                </td>
            </tr>
        `).join('');

        console.log(`Обновление таблицы с результатами (${this.currentResults.length} записей).`);

        this.elements.resultsTable.innerHTML = tableRows;
    }

    showNotification(message, type) {
        const notification = document.createElement('div');

        notification.className = `notification notification-${type} animate__animated animate__fadeInRight`;

        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;

        document.body.appendChild(notification);

        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.classList.add('animate__fadeOutRight');

            setTimeout(() => notification.remove(), 300);
        });

        setTimeout(() => {
            if (notification.parentElement) {
                notification.classList.add('animate__fadeOutRight');

                setTimeout(() => notification.remove(), 300);
            }
        }, 3000);
    }

}

document.addEventListener('DOMContentLoaded', () => {
    new MedAnalytics();
});