if not exists(select * from sys.databases where name = 'lpdb') begin create database lpdb end
go
use lpdb

-- license plate is 10, even though no state that I know of has license plates longer than 8 characters, 
-- just to give it some wiggle room
if not exists(select 1 from sys.tables where name = 'main_gate_alpr_license_plates' and type = 'U') begin
    create table main_gate_alpr_license_plates (
        -- filled out by the server
        id uniqueidentifier primary key,
        license_plate nvarchar(10) not null,
        captured_at datetime not null,

        -- filled out by the user
        license_plate_corrected nvarchar(10),
        visitor_name nvarchar(50),
        visitor_company_name nvarchar(50),
        visitor_receiver_name nvarchar(50),
        visit_reason nvarchar(200),
        soft_deleted bit not null default 0,
    ) 
end
go