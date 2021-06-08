--CARA INSTALL APLIKASI--
Sebelum melakukan langkah dibawah ini, Pastikan sudah menginstall python
1. Download file aplikasi dengan format ZIP atau bisa juga dengan clone repository
2. Ekstrak file ZIP
3. Setelah itu arahkan ke direktori aplikasi yang telah di ekstrak
4. Buka cmd atau git bash dan jalankan perintah dilangkah selanjutnya secara bertahap
5. Membuat folder virtual envirorment : python -m venv venv
6. Lalu masuk ke dalam folder virtual envirorment : cd venv
7. Install requirement aplikasi : pip install -r requirements.txt
8. Lalu keluar dari virtual envirorment : cd ..
9. Pindah direktori ke codeApp untuk menjalankan aplikasi : cd codeApp
10. Jalankan aplikasi : python manage.py runserver
11. Setelah server berjalan copy url server ke browser: http://127.0.0.1:8000/
12. Tambahkan url halaman dibelakang url server, berikut adalah url halaman yang tersedia :
    - Halaman training metode : http://127.0.0.1:8000/index
    - Halaman testing metode : http://127.0.0.1:8000/test
    - Halaman training metode untuk end user : http://127.0.0.1:8000/user
13. Jika ingin menghentikan server cukup tekan Ctrl + C pada cmd yang digunakan untuk running server
