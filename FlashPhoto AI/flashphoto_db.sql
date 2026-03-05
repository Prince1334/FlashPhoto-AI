-- ============================================================
--  FlashPhoto AI  –  MySQL Database Setup
--  Database : flashphoto_db
--  Default credentials (match app/config.py defaults):
--    host     : localhost
--    user     : root
--    password : root
--
--  Run with:
--    mysql -u root -p < flashphoto_db.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS `flashphoto_db`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE `flashphoto_db`;

-- ------------------------------------------------------------
-- Table: events
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `events` (
    `id`          INT            NOT NULL AUTO_INCREMENT,
    `name`        VARCHAR(255)   NOT NULL,
    `access_code` VARCHAR(255)   NOT NULL,
    `created_at`  DATETIME       DEFAULT NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_events_access_code` (`access_code`)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- Table: photos
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `photos` (
    `id`        INT          NOT NULL AUTO_INCREMENT,
    `event_id`  INT          NOT NULL,
    `file_path` VARCHAR(255) NOT NULL,
    PRIMARY KEY (`id`),
    CONSTRAINT `fk_photos_event`
        FOREIGN KEY (`event_id`)
        REFERENCES `events` (`id`)
        ON DELETE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- Table: face_encodings
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `face_encodings` (
    `id`            INT      NOT NULL AUTO_INCREMENT,
    `photo_id`      INT      NOT NULL,
    `encoding_json` LONGTEXT NOT NULL,
    PRIMARY KEY (`id`),
    CONSTRAINT `fk_encodings_photo`
        FOREIGN KEY (`photo_id`)
        REFERENCES `photos` (`id`)
        ON DELETE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- Alembic migration tracking table
-- (needed so Flask-Migrate knows the schema is up to date)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `alembic_version` (
    `version_num` VARCHAR(32) NOT NULL,
    PRIMARY KEY (`version_num`)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;

-- Stamp the initial migration revision
INSERT IGNORE INTO `alembic_version` (`version_num`)
VALUES ('aea7e68d5a9a');
