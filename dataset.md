[Датасет](https://drive.google.com/file/d/1AnXq1ZxM3Yo7Tz6r0TnpnTPmOfcyjgMX/view?usp=drive_link)

Основные признаки:
* Описание вакансии
  * `name` - название вакансии
  * `description` - текстовое описание вакансии на HeadHunter
  * `schedule` - тип рабочего графика
  * `professional_roles_name` - профессиональная категория согласно HeadHunter
  * `published_at` - дата публикации вакансии
* Зарплата
  * `salary_from` - нижняя граница вилки зарплаты
  * `salary_to` - верхняя граница вилки зарплаты
  * `salary_gross` - индикатор, если зарплата указана в размере gross
  * `salary_currency` - валюта зарплаты
* Требования к кандидату
  * `experience` - требуемый опыт для вакансии
  * `key_skills` - требуемые навыки
  * `languages` - требуемое владение иностранными языками
* Работодатель
  * `employer_name` - название работодателя
  * `accredited_it_employer` - индикатор для аккредитованных в России IT-компаний
* Место работы
  * `area_name` - названия населенного пункта, в котором размещена вакансия
  * `addres_raw` - адрес места работы
  * `addres_lat` - широта места работы
  * `address_lng` - долгота места работы
