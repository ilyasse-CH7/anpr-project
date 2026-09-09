-- Schéma minimal de stockage des matricules ANPR (MySQL).
--
-- À exécuter une fois avec un compte administrateur :
--     sudo mysql < config/schema_mysql.sql
-- puis renseigner ANPR_DB_PASSWORD dans le fichier .env.

CREATE DATABASE IF NOT EXISTS anpr
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Remplacer 'CHANGEME' par un mot de passe réel avant exécution.
CREATE USER IF NOT EXISTS 'anpr'@'localhost' IDENTIFIED BY 'CHANGEME';
GRANT SELECT, INSERT, UPDATE, DELETE ON anpr.* TO 'anpr'@'localhost';
FLUSH PRIVILEGES;

USE anpr;

CREATE TABLE IF NOT EXISTS plates (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    matricule          VARCHAR(32)  NOT NULL,
    confiance          FLOAT        NOT NULL DEFAULT 0,
    timestamp          DATETIME     NOT NULL,
    chemin_image_crop  VARCHAR(512) NULL,
    INDEX idx_plates_timestamp (timestamp),
    INDEX idx_plates_matricule (matricule)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
