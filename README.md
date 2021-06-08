--CARA INSTALL APLIKASI--
Sebelum melakukan langkah dibawah ini, Pastikan sudah menginstall python
1. Download file aplikasi dengan format ZIP atau bisa juga dengan clone repository
2. Ekstrak file ZIP
3. Setelah itu arahkan ke direktori aplikasi yang telah di ekstrak
4. Buka cmd atau git bash dan jalankan perintah berikut secara bertahap :
    a. Membuat folder virtual envirorment : python -m venv venv
    b. Lalu masuk ke dalam folder virtual envirorment : cd venv
    c. Install requirement aplikasi : pip install -r requirements.txt
    d. Lalu keluar dari virtual envirorment : cd ..
    e. Pindah direktori ke codeApp untuk menjalankan aplikasi : cd codeApp
    f. Jalankan aplikasi : python manage.py runserver
    g. Setelah server berjalan copy url server ke browser: http://127.0.0.1:8000/
    h. Tambahkan url halaman dibelakang url server, berikut adalah url halaman yang tersedia :
        - Halaman training metode : http://127.0.0.1:8000/index
        - Halaman testing metode : http://127.0.0.1:8000/test
        - Halaman training metode untuk end user : http://127.0.0.1:8000/user
    i. Jika ingin menghentikan server cukup tekan Ctrl + C pada cmd yang digunakan untuk running server
