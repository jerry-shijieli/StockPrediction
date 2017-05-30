create table itDays 
(select distinct timestamp as tDay from intStocks  order by tDay asc);

alter table itDays add id int PRIMARY key aUTO_INCREMENT;

create table itradingDays 
select a.id as id, a.tDay as tDay, b.tDay as prevtDay from
(select id, tDay from itDays) a
left join 
(select id+1 as id, tDay from itDays) b
on a.id=b.id;


alter table itradingDays 
add column year int,
add column month int,
add column day int,
add column dayOfWeek int, 
add column tDaysleftMonth int,
add column tDaysleftWeek int;

alter table itradingDays 
add column weekOfYear int;

alter table itradingDays 
add column tDayinMonth int;

alter table itradingDays 
add column tDayinWeek int;

update itradingDays set year=year(tDay);

update itradingDays set month=month(tDay);

update itradingDays set day=day(tDay);

update itradingDays set dayOfWeek=weekday(tDay);

update itradingDays set weekOfYear=week(tDay);


create table imaxmintDays (select year, month, min(id) as mintDay, max(id) as maxtDay from itradingDays
group by year,month);

update itradingDays set tDaysleftMonth =(select maxtDay from 
imaxmintDays where itradingDays.year=imaxmintDays.year and itradingDays.month=imaxmintDays.month)-id;
#(select quantity from table_1 where locationid=1 and table_1.itemid = table_2.itemid)

update itradingDays set tDayinMonth = id+1-(select mintDay from 
imaxmintDays where itradingDays.year=imaxmintDays.year
 and itradingDays.month=imaxmintDays.month);

create table itWeeks (select year, weekOfYear, min(id) as mintDay,
max(id) as maxtDay from itradingDays
 group by year,weekOfYear) ;

update itradingDays set tDaysleftWeek =  (select maxtDay from 
itWeeks where itradingDays.year=itWeeks.year and itradingDays.weekOfYear=itWeeks.weekOfYear)-id;


update itradingDays set tDayinWeek = id+1-(select mintDay from 
itWeeks where itradingDays.year=itWeeks.year and 
itradingDays.weekOfYear=
itWeeks.weekOfYear);
