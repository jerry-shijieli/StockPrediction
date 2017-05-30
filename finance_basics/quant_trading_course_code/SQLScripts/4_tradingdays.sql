create table tDays 
(select distinct timestamp as tDay from indices where name="NIFTY"
  order by tDay asc);

alter table tDays add id int PRIMARY key aUTO_INCREMENT;

create table tradingDays 
select a.id as id, a.tDay as tDay, b.tDay as prevtDay from
(select id, tDay from tDays) a
left join 
(select id+1 as id, tDay from tDays) b
on a.id=b.id;


alter table tradingDays 
add column year int,
add column month int,
add column day int,
add column dayOfWeek int, 
add column tDaysleftMonth int,
add column tDaysleftWeek int,
add column weekOfYear int,
add column tDayinMonth int,
add column tDayinWeek int;

update tradingDays set year=year(tDay);

update tradingDays set month=month(tDay);

update tradingDays set day=day(tDay);

update tradingDays set dayOfWeek=weekday(tDay);

update tradingDays set weekOfYear=week(tDay);


create table maxmintDays (select year, month, min(id) as mintDay,
 max(id) as maxtDay from tradingDays
group by year,month);

update tradingDays set tDaysleftMonth =(select maxtDay from 
maxmintDays where tradingDays.year=maxmintDays.year and tradingDays.month=maxmintDays.month)-id;


update tradingDays set tDayinMonth = id+1-(select mintDay from 
imaxmintDays where tradingDays.year=maxmintDays.year
 and tradingDays.month=maxmintDays.month);

create table tWeeks (select year, weekOfYear, min(id) as mintDay,
max(id) as maxtDay from tradingDays
 group by year,weekOfYear) ;

update tradingDays set tDaysleftWeek =  (select maxtDay from 
tWeeks where tradingDays.year=tWeeks.year and tradingDays.weekOfYear=tWeeks.weekOfYear)-id;


update tradingDays set tDayinWeek = id+1-(select mintDay from 
tWeeks where tradingDays.year=tWeeks.year and 
tradingDays.weekOfYear=
tWeeks.weekOfYear);
